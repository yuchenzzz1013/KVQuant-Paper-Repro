from typing import List

from torch import nn

_PATCHED: List = []


def patch_model(model: nn.Module, quantizer):
    """
    把 KVQuantizer 挂到每层的 k_proj / v_proj 上。
    Pre-RoPE Key 量化 = hook 在 k_proj 之后、RoPE 之前。
    """
    global _PATCHED
    unpatch_model(model)  # 先清掉旧的

    hooks = []
    for i, layer in enumerate(model.model.layers):
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj

        def make_k_hook(idx):
            def hook(module, inp, out):
                return quantizer.quantize_k(idx, out)
            return hook

        def make_v_hook(idx):
            def hook(module, inp, out):
                return quantizer.quantize_v(idx, out)
            return hook

        hooks.append(k_proj.register_forward_hook(make_k_hook(i)))
        hooks.append(v_proj.register_forward_hook(make_v_hook(i)))

    _PATCHED = hooks
    return hooks


def unpatch_model(model: nn.Module = None):
    global _PATCHED
    for h in _PATCHED:
        try:
            h.remove()
        except Exception:
            pass
    _PATCHED = []