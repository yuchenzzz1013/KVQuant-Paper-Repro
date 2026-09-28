from typing import List
from torch import nn

_PATCHED: List = []


def patch_model(model: nn.Module, quantizer):
    """把 KVQuantizer 挂到每层的 k_proj / v_proj。

    - K hook 在 RoPE 之前（Pre-RoPE Key 量化，论文 §3.2）
    - V hook 保持 per-token 量化
    - 仅做反量化替换，不做真实内存压缩（PPL 复现用）
    """
    global _PATCHED
    unpatch_model(model)

    hooks = []
    for i, layer in enumerate(model.model.layers):
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj

        def make_k_hook(idx):
            def hook(module, inp, out):
                assert out.dim() == 3, f"k_proj 输出维度异常: {out.shape}"
                return quantizer.quantize_k(idx, out)
            return hook

        def make_v_hook(idx):
            def hook(module, inp, out):
                assert out.dim() == 3, f"v_proj 输出维度异常: {out.shape}"
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