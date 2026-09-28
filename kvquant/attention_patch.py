from typing import List
from torch import nn

_PATCHED: List = []


def patch_model(model: nn.Module, quantizer):
    """把 KVQuantizer 挂到每层的 k_proj / v_proj。

    注意：hook 直接返回反量化后的 K/V，
    - K 发生在 RoPE 之前（Pre-RoPE Key 量化，论文 3.2）
    - V 保持 per-token 量化
    这是为 PPL 复现实验设计的轻量实现。
    """
    global _PATCHED
    unpatch_model(model)

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