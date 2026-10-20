"""Generate the cover: the reference tokens as forced history, fresh CFG-1 AR over the composite score, v9 NAR at 64 steps."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import uuid


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_inputs(config, request, reference):
    if (request.get("cot") != "full" or not isinstance(request.get("abc"), str) or not request["abc"].strip()
            or request.get("seed") != 42 or request.get("cfg_scale") != 1.0):
        raise ValueError("Require this song's complete ABC, cot=full, seed=42 and cfg_scale=1")
    if (reference.ndim != 1 or reference.dtype.kind not in "iu" or
            len(reference) != config["reference_frames"] or not len(reference) or
            reference.min() < 0 or reference.max() >= 32768):
        raise ValueError("Reference must contain the configured number of raw codec frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=["generate"])
    args = parser.parse_args()
    config = read(args.config)
    request = read(config["request_path"])
    if sha(config["nar_path"]) != config["nar_sha256"]:
        raise ValueError("NAR checkpoint hash mismatch")
    import numpy as np
    from yue2.protocol import GenerationConfig, SongRequest, resolve_sampling
    from yue2.storage import identity, verify_result, write_json
    reference = np.load(config["semantic_prefix_path"], allow_pickle=False)
    validate_inputs(config, request, reference)
    request = SongRequest(**request).to_dict()
    forced = reference.tolist()
    sampling = resolve_sampling(config["semantic_sampling"], GenerationConfig().semantic)
    if sampling.max_tokens <= len(forced):
        raise ValueError("Total semantic budget must leave room for this song after the reference")
    provenance = {
        "phase": args.phase, "arm": "voice-prefix", "request": request,
        "semantic_sampling": dataclasses.asdict(sampling), "forced_frames": len(forced),
        "reference_sha256": sha(config["semantic_prefix_path"]), "nar_sha256": config["nar_sha256"],
        "cfg_scale": 1.0, "ode_steps": 64, "ar_regenerated": True, "ar_lora": False, "svc_used": False,
        "script_sha256": sha(__file__), "prefix_adapter_sha256": sha(Path(__file__).with_name("prefix_sampling.py")),
        "nar_merge_sha256": sha(Path(__file__).with_name("nar.py")),
    }
    contract = identity(provenance)
    output = Path(config["output_root"]) / "voice-prefix"
    output.mkdir(parents=True, exist_ok=True)
    complete = output / "complete.json"
    if complete.exists():
        old = read(complete)
        if old["contract"] != contract or identity(old["provenance"]) != contract:
            raise ValueError("Completed song contract changed; use a new output directory")
        root = Path(old["artifact_root"])
        receipt = verify_result(root, expected_identity=old["result"]["identity"])
        if (receipt != old["result"] or sha(root / "audio.flac") != old["audio_sha256"] or
                identity({"request": read(root / "request.json"), "config": read(root / "config.json"),
                          "weights": receipt["weights"]}) != receipt["identity"]):
            raise ValueError("Completed song identity or receipt changed")
        print(json.dumps(old), flush=True)
        return
    attempt = output / ("attempt-" + uuid.uuid4().hex[:12])
    attempt.mkdir()
    write_json(attempt / "request-contract.json", provenance)
    (attempt / "pid").write_text(str(os.getpid()), encoding="utf-8")
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    start = time.perf_counter()
    import torch
    from ckpt_io import load_ckpt
    from yue2 import YuE2Pipeline
    from nar import merge_nar
    from prefix_sampling import generate_tokens_with_prefix
    torch.cuda.reset_peak_memory_stats()
    try:
        with YuE2Pipeline.from_pretrained(config["model_dir"], vae=config["vae_dir"], device="cuda",
                                         quantization="none", local_files_only=True) as pipe:
            pipe.generation_config = dataclasses.replace(pipe.generation_config, ode_steps=64)
            model = pipe._load_model()
            nar = load_ckpt(config["nar_path"], "cpu")
            merge_nar(model, nar, torch)
            del nar

            # This instance always performs new AR; no previous song semantic is loaded.
            def generate(prefix, sampling, seed, phase, **kwargs):
                if phase != "semantic":
                    raise ValueError("Use the provided complete score; no planner generation")
                return generate_tokens_with_prefix(model, prefix, sampling, seed, phase,
                                                   semantic_prefix=forced, use_cuda_graph=True, **kwargs)

            pipe._generate = generate
            count = [0]

            def progress(phase, token):
                count[0] += 1
                if count[0] % 250 == 0:
                    print(json.dumps({"event": "tokens", "sampled": count[0], "forced": len(forced),
                                      "seconds": time.perf_counter() - start}), flush=True)

            result = pipe(**request, semantic_sampling=config["semantic_sampling"],
                          cancelled=lambda: stopped[0], on_token=progress)
            if stopped[0]:
                raise InterruptedError("Cancelled after audio decode")
            tokens = result.semantic.tokens
            if tokens[:len(forced)] != forced or not len(forced) < len(tokens) <= sampling.max_tokens:
                raise ValueError("Expected unchanged reference plus new semantic frames within the total budget")
            if result.config["generation"]["ode_steps"] != 64 or result.config["cfg_scale"] != 1.0:
                raise ValueError("Effective synthesis settings differ from CFG 1 / 64 steps")
            result.config["voice_prefix"] = provenance
            result.request_identity = identity({"request": result.semantic.plan.request.to_dict(),
                                                "config": result.config, "weights": result.weights})
            receipt = result.save_artifacts(attempt / "result")
            verify_result(attempt / "result", expected_identity=result.request_identity)
        if stopped[0]:
            raise InterruptedError("Cancelled before completion marker")
        record = {
            "contract": contract, "provenance": provenance, "phase": args.phase, "arm": "voice-prefix",
            "case": "reference-continuation", "label": "voice-prefix", "cfg_scale": 1.0, "ode_steps": 64,
            "ar_regenerated": True, "elapsed_seconds": time.perf_counter() - start, "result": receipt,
            "artifact_root": str(attempt / "result"), "audio_sha256": sha(attempt / "result/audio.flac"),
            "semantic_sha256": sha(attempt / "result/semantic.npy"), "prefix_sha256": sha(attempt / "result/prefix.npy"),
            "duration_seconds": receipt["audio_seconds"], "forced_seconds": len(forced) / 25,
            "forced_frames": len(forced), "new_frames": len(tokens) - len(forced), "truncated": result.semantic.truncated,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "svc_used": False,
        }
        write_json(complete, record)
        print(json.dumps(record), flush=True)
    except InterruptedError:
        write_json(attempt / "interrupted.json", {"elapsed_seconds": time.perf_counter() - start})
        raise SystemExit(75)


if __name__ == "__main__":
    main()
