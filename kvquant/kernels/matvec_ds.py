import torch


def dense_matvec_lut(x: torch.Tensor, codebook: torch.Tensor):
    """
    把 x 先 LUT 量化再线性还原（PyTorch 参考实现）。

    x:        [..., D]
    codebook: [2^b]
    返回:     [..., D]，即 dequant(x)
    """
    flat = x.reshape(-1, 1).float()
    cb = codebook.reshape(1, -1).to(x.device).float()
    idx = (flat - cb).abs().argmin(dim=-1)
    return cb[0][idx].reshape(x.shape).to(x.dtype)


def sparse_matvec(
    indices: torch.Tensor,
    values: torch.Tensor,
    shape,
    x: torch.Tensor,
):
    """
    稀疏矩阵 × 向量，PyTorch 参考实现。

    indices: [nnz, 2]，每行 (row, col)
    values:  [nnz]
    shape:   (R, C)
    x:       [C]
    返回:    [R]
    """
    idx = indices.t().long()
    sp = torch.sparse_coo_tensor(idx, values, size=shape, device=x.device)
    return torch.sparse.mm(sp, x.unsqueeze(1)).squeeze(1)