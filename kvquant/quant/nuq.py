import torch
import numpy as np
from sklearn.cluster import KMeans


def train_nuq_codebook(activations: torch.Tensor, fisher: torch.Tensor, n_bits: int = 3):
    """
    activations: [N] 已归一化到 [-1, 1]
    fisher:      [N] 敏感度权重
    返回:         [2^n_bits] 排序后的码本中心
    """
    n_clusters = 2 ** n_bits
    X = activations.reshape(-1, 1).cpu().numpy()
    w = fisher.reshape(-1).cpu().numpy()
    w = w / (w.sum() + 1e-8)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=100, random_state=0)
    kmeans.fit(X, sample_weight=w)
    centers = torch.tensor(kmeans.cluster_centers_.flatten(), dtype=torch.float32)
    centers = torch.sort(centers).values
    return centers