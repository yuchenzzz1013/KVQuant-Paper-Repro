import torch


class QuantizedKVCache:
    """
    存量化后的 K/V：
        - K: uint8 indices，per-channel scale / zero
        - V: uint8 indices，per-token scale / zero
        - 稀疏 outlier 以 CSR 保存
    """

    def __init__(self):
        self.k_cache = []   # list[dict]
        self.v_cache = []
        self.seq_len = 0

    def append_k(self, layer_idx: int, packed_k: dict):
        while len(self.k_cache) <= layer_idx:
            self.k_cache.append(None)
        self.k_cache[layer_idx] = packed_k

    def append_v(self, layer_idx: int, packed_v: dict):
        while len(self.v_cache) <= layer_idx:
            self.v_cache.append(None)
        self.v_cache[layer_idx] = packed_v

    def clear(self):
        self.k_cache = []
        self.v_cache = []
        self.seq_len = 0

    def memory_bytes(self) -> int:
        total = 0
        for cache in (self.k_cache, self.v_cache):
            for packed in cache:
                if packed is None:
                    continue
                total += packed["q_idx"].numel()
                total += packed["outlier_values"].numel() * 2
                total += packed["outlier_cols"].numel() * 2
                total += packed["outlier_rowptr"].numel() * 4
                total += packed["scale"].numel() * 2
                total += packed["zero"].numel() * 2
        return total