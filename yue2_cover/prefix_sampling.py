"""YuE2 semantic sampling with a forced history (the singer's reference tokens).

Sampling loop follows the installed yue2/sampling.py. The added history follows
audio.cpp f1c299b73c06df27458e2ea8a7ab6638891ea3ba: append forced tokens to both
CFG branches, retain them in the returned stream, and count them in bounds and
the repetition window. This module does not modify the installed package.
"""
from __future__ import annotations

from numbers import Integral
import time

import torch

from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END
from yue2.sampling import distribution, synchronize


@torch.inference_mode()
def generate_tokens_with_prefix(model, prefix, sampling, seed, phase="semantic", *,
                                semantic_prefix=(), negative=None, cfg_scale=1.0,
                                legacy_off=False, cancelled=None, on_token=None,
                                use_cuda_graph=True):
    """Return absolute token IDs, timing, truncated; token bounds include prefix.

    ``semantic_prefix`` contains raw codec indices, never model vocabulary IDs.
    ``prefix`` and ``negative`` retain their ordinary text/music-start tokens.
    The caller must keep SymbolicPlan.prefix unchanged for later NAR synthesis.
    """
    if phase != "semantic":
        raise ValueError("Forced audio history is supported only for semantic generation")
    codec = list(semantic_prefix)
    if any(isinstance(t, bool) or not isinstance(t, Integral) or not 0 <= t < CODEC_SIZE for t in codec):
        raise ValueError("semantic_prefix requires raw integer codec indices in [0,32768)")
    forced = [int(t) + CODEC_OFFSET for t in codec]
    forced_count = len(forced)
    if forced_count > sampling.max_tokens:
        raise ValueError("semantic_prefix exceeds total max_tokens")
    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    if negative is not None and len(negative) + sampling.max_tokens > CONTEXT:
        raise ValueError("Negative prefix + generation budget exceeds context")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")
    remaining = sampling.max_tokens - forced_count
    if remaining == 0:
        return forced, {"seconds": 0.0, "prefill_seconds": 0.0, "ttft_seconds": None,
                        "output_tokens": 0, "content_tokens": forced_count,
                        "forced_tokens": forced_count, "sampled_content_tokens": 0,
                        "output_tps": 0.0, "prefix_tokens": len(prefix),
                        "cfg_branches": 0, "execution": "forced_only"}, True

    from yue2.modeling_yue2 import StaticKVCache

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prompt_count = len(prefix)
    positive_ids = list(prefix) + forced
    negative_ids = None if negative is None else list(negative) + forced
    rng_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=rng_device).manual_seed(seed)
    config = model.config

    def prefill(ids):
        cache = StaticKVCache(num_layers=config.num_hidden_layers, batch_size=1,
                              num_kv_heads=config.num_key_value_heads,
                              max_seq_len=len(ids) + remaining,
                              head_dim=config.head_dim, dtype=dtype, device=device)
        output = model(torch.tensor([ids], device=device), past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        return output.logits[:, -1, :], output.past_key_values

    graph = None
    positive_cache = negative_cache = None
    graph_enabled = use_cuda_graph and device.type == "cuda" and not getattr(model, "_yue2_fp8_originals", {})
    synchronize(device)
    start = time.perf_counter()
    try:
        if graph_enabled:
            from yue2.cuda_graph import GraphAR
            graph = GraphAR(model, [positive_ids] if cfg_scale == 1 else [positive_ids, negative_ids], remaining)
            logits = graph.prefill()
            conditional = logits[:1]
            unconditional = logits[1:] if cfg_scale != 1 else None
        else:
            conditional, positive_cache = prefill(positive_ids)
            unconditional = None
            if cfg_scale != 1.0:
                unconditional, negative_cache = prefill(negative_ids)
        synchronize(device)
        prefill_seconds = time.perf_counter() - start
        history, first, eos = list(forced), None, False
        for step in range(forced_count, sampling.max_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during semantic")
            # Preserve the installed BF16 CFG arithmetic before distribution upcast.
            logits = conditional if cfg_scale == 1.0 else unconditional + cfg_scale * (conditional - unconditional)
            scores = distribution(logits, sampling, history, step, phase, legacy_off)
            if sampling.temperature == 0:
                next_id = scores.argmax(-1, keepdim=True)
            else:
                probabilities = scores.softmax(-1)
                if device.type == "mps":
                    next_id = torch.multinomial(probabilities.cpu(), 1, generator=generator).to(device)
                else:
                    next_id = torch.multinomial(probabilities, 1, generator=generator)
            token = int(next_id.item())
            if first is None:
                first = time.perf_counter() - start
            if on_token is not None:
                on_token(phase, token)
            if token == MUSIC_END:
                eos = True
                break
            history.append(token)
            if step + 1 < sampling.max_tokens:
                if graph is not None:
                    branch_logits = graph.step(next_id)
                    conditional = branch_logits[:1]
                    unconditional = branch_logits[1:] if cfg_scale != 1 else None
                else:
                    conditional = model(next_id, past_key_values=positive_cache, use_cache=True,
                                        logits_to_keep=1).logits[:, -1, :]
                    if negative_cache is not None:
                        unconditional = model(next_id, past_key_values=negative_cache, use_cache=True,
                                              logits_to_keep=1).logits[:, -1, :]
        synchronize(device)
        seconds = time.perf_counter() - start
        sampled_content = len(history) - forced_count
        count = sampled_content + int(eos)
        timing = {"seconds": seconds, "prefill_seconds": prefill_seconds,
                  "ttft_seconds": first, "output_tokens": count, "content_tokens": len(history),
                  "forced_tokens": forced_count, "sampled_content_tokens": sampled_content,
                  "output_tps": count / seconds, "prefix_tokens": prompt_count,
                  "prefill_tokens": len(positive_ids),
                  "cfg_branches": 1 if cfg_scale == 1 else 2,
                  "execution": "cuda_graph" if graph is not None else "eager",
                  "attention": graph.attention_backend if graph is not None else "sdpa"}
        if use_cuda_graph and not graph_enabled:
            timing["graph_fallback_reason"] = "fp8_not_graph_validated" if getattr(model, "_yue2_fp8_originals", {}) else "non_cuda_device"
        return history, timing, not eos
    finally:
        if graph is not None:
            graph.close()
        positive_cache = negative_cache = None
