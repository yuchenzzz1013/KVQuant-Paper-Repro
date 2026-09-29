import torch

from .cuda_ops import get_cuda_ops


def dense_matvec_lut(x: torch.Tensor, codebook: torch.Tensor):
    """LUT 最近邻反量化（CUDA 优先）。

    x:        [..., D]
    codebook: [2^b]
    返回:     [..., D]
    """
    cuda_ops = get_cuda_ops()
    if cuda_ops is not None and x.is_cuda:
        cb = codebook.to(x.device, dtype=torch.float32).reshape(-1).contiguous()
        _, dq = cuda_ops.lut_lookup(x.reshape(-1).float(), cb)
        return dq.reshape(x.shape).to(x.dtype)

    # PyTorch 回退
    flat = x.reshape(-1, 1).float()
    cb = codebook.reshape(1, -1).to(x.device).float()
    idx = (flat - cb).abs().argmin(dim=-1)
    return cb[0][idx].reshape(x.shape).to(x.dtype)


def sparse_matvec_csr(row_ptr, col_idx, values, x):
    """CSR 稀疏矩阵 × 向量（CUDA 优先）。

    row_ptr: [R+1] int32
    col_idx: [nnz] int32
    values:  [nnz] fp16
    x:       [C]  fp16
    返回:    [R]  fp16
    """
    cuda_ops = get_cuda_ops()
    if cuda_ops is not None and x.is_cuda:
        R = int(row_ptr.numel()) - 1
        return cuda_ops.sparse_matvec_csr(row_ptr, col_idx, values, x, R)

    # PyTorch 回退
    R = int(row_ptr.numel()) - 1
    out = torch.zeros(R, dtype=x.dtype, device=x.device)
    rp = row_ptr.tolist()
    for r in range(R):
        s, e = rp[r], rp[r + 1]
        if e > s:
            out[r] = (values[s:e].float()
                      * x[col_idx[s:e].long()].float()).sum().to(x.dtype)
    return out


def sparse_matvec(indices: torch.Tensor, values: torch.Tensor,
                  shape, x: torch.Tensor):
    """COO 稀疏矩阵 × 向量（保留原接口，内部转换为 CSR 走 CUDA）。"""
    cuda_ops = get_cuda_ops()
    if cuda_ops is not None and x.is_cuda:
        R, C = shape
        rows = indices[:, 0].long()
        cols = indices[:, 1].long()
        order = torch.argsort(rows, stable=True)
        rows_s, cols_s, vals_s = rows[order], cols[order], values[order]

        row_counts = torch.bincount(rows_s, minlength=R)
        row_ptr = torch.zeros(R + 1, dtype=torch.int32, device=x.device)
        row_ptr[1:] = row_counts.cumsum(0).to(torch.int32)
        return cuda_ops.sparse_matvec_csr(
            row_ptr, cols_s.to(torch.int32), vals_s.contiguous(), x, R)

    # PyTorch 回退
    idx = indices.t().long()
    sp = torch.sparse_coo_tensor(idx, values, size=shape, device=x.device)
    return torch.sparse.mm(sp, x.unsqueeze(1)).squeeze(1)