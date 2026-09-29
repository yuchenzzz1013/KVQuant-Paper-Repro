"""CUDA 算子加载器：首次使用时 JIT 编译，失败则回退 PyTorch。

编译好的扩展缓存在 ~/.cache/torch_extensions/ 下，重启进程后可复用。
"""
from pathlib import Path

import torch

_cuda_ops = None
_load_error = None


def _try_load():
    global _cuda_ops, _load_error
    if _cuda_ops is not None:
        return _cuda_ops
    if _load_error is not None:
        return None
    if not torch.cuda.is_available():
        _load_error = "CUDA not available"
        return None
    src = Path(__file__).resolve().parent / "csrc" / "kvquant_ops.cu"
    if not src.exists():
        _load_error = f"CUDA source not found: {src}"
        return None
    try:
        from torch.utils.cpp_extension import load
        _cuda_ops = load(
            name="kvquant_cuda",
            sources=[str(src)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
            verbose=False,
        )
        return _cuda_ops
    except Exception as e:
        _load_error = repr(e)
        return None


def get_cuda_ops():
    """返回已加载的 CUDA 扩展（或 None）。"""
    return _try_load()


def cuda_available() -> bool:
    return _try_load() is not None