"""Merge the community joint-v9 NAR LoRA into YuE2's NAR branch (it pairs with the v9 tokenizer head)."""


def merge_nar(model, nar, torch):
    """nar: ckpt_io.load_ckpt() of nar_lora_joint_v9 -> {'lora': [A0, B0, ...], 'io': {...}}; merged in place."""
    pairs = iter(nar['lora'])
    with torch.no_grad():
        for layer in model.model.layers:
            for module, names in ((layer.nar_self_attn, ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
                                  (layer.nar_mlp, ('gate_proj', 'up_proj', 'down_proj'))):
                for name in names:
                    a, b = next(pairs), next(pairs)
                    linear = getattr(module, name)
                    delta = b.to(linear.weight.device).float() @ a.to(linear.weight.device).float()
                    if delta.shape != linear.weight.shape:
                        raise ValueError('NAR_LORA_SHAPE_MISMATCH')
                    linear.weight.add_(delta.to(linear.weight.dtype))
        if next(pairs, None) is not None:
            raise ValueError('NAR_LORA_HAS_EXTRA_TENSORS')
        model.vae2llm.load_state_dict(nar['io']['vae2llm'])
        model.llm2vae.load_state_dict(nar['io']['llm2vae'])
