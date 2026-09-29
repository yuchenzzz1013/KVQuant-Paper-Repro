import copy
import math

import torch
from tqdm import tqdm

from kvquant.quant.nuq import train_nuq_codebook
from kvquant.quant.dense_sparse import extract_outliers
from kvquant.quant.kv_quantizer import KVQuantizer


@torch.no_grad()
def estimate_layer_sensitivity(
    model,
    tokenizer,
    calib_texts,
    seq_len: int = 2048,
    device: str = "cuda",
    candidate_bits: list = None,
    outlier_ratio: float = 0.01,
    sink_token: bool = True,
):
    """逐层敏感度估计（论文 §3.6）。

    对每一层，分别用 candidate_bits 中的比特宽度模拟量化，
    测量量化后该层对模型损失的扰动。

    返回:
        sensitivity: dict, {layer_idx: {bit_width: loss_delta}}
    """
    if candidate_bits is None:
        candidate_bits = [2, 3, 4]

    model.eval()
    n_layers = len(model.model.layers)

    # ---------- 1. 收集每层的 K/V 激活和 Fisher 信息 ----------
    act_store, grad_store = {}, {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj

        def make_act_hook(name):
            def hook(module, inp, out):
                act_store[name] = out.detach()
            return hook

        def make_grad_hook(name):
            def hook(module, grad_in, grad_out):
                grad_store[name] = grad_out[0].detach()
            return hook

        hooks.append(k_proj.register_forward_hook(make_act_hook(f"k_{i}")))
        hooks.append(v_proj.register_forward_hook(make_act_hook(f"v_{i}")))
        hooks.append(k_proj.register_full_backward_hook(make_grad_hook(f"k_{i}")))
        hooks.append(v_proj.register_full_backward_hook(make_grad_hook(f"v_{i}")))

    k_acts = {i: [] for i in range(n_layers)}
    v_acts = {i: [] for i in range(n_layers)}
    fisher_k = {i: None for i in range(n_layers)}
    fisher_v = {i: None for i in range(n_layers)}

    for text in tqdm(calib_texts, desc="Estimating sensitivity"):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=seq_len
        ).to(device)
        labels = inputs["input_ids"].clone()

        model.zero_grad(set_to_none=True)
        outputs = model(**inputs, labels=labels)
        outputs.loss.backward()

        for i in range(n_layers):
            k_act = act_store.get(f"k_{i}")
            v_act = act_store.get(f"v_{i}")
            k_grad = grad_store.get(f"k_{i}")
            v_grad = grad_store.get(f"v_{i}")

            if k_act is not None and k_grad is not None:
                if sink_token and k_act.size(1) > 1:
                    k_act = k_act[:, 1:, :]
                    k_grad = k_grad[:, 1:, :]
                k_acts[i].append(k_act.float().cpu())
                g = (k_grad.float() ** 2).sum(dim=(0, 1)).cpu()
                fisher_k[i] = g if fisher_k[i] is None else fisher_k[i] + g

            if v_act is not None and v_grad is not None:
                if sink_token and v_act.size(1) > 1:
                    v_act = v_act[:, 1:, :]
                    v_grad = v_grad[:, 1:, :]
                v_acts[i].append(v_act.float().cpu())
                g = (v_grad.float() ** 2).sum(dim=(0, 1)).cpu()
                fisher_v[i] = g if fisher_v[i] is None else fisher_v[i] + g

        act_store.clear()
        grad_store.clear()

    for h in hooks:
        h.remove()

    # ---------- 2. 对每层、每比特宽度模拟量化，测量损失扰动 ----------
    sensitivity = {i: {} for i in range(n_layers)}

    # 先用 fp16 跑一遍基线 loss
    baseline_loss = _compute_loss(model, tokenizer, calib_texts, seq_len, device)

    for bit in candidate_bits:
        # 用该比特宽度训练码本（仅用于敏感度估计的临时量化器）
        k_codebooks, v_codebooks = [], []
        k_scales, k_zeros = [], []

        for i in range(n_layers):
            k_cat = torch.cat(k_acts[i], dim=0)
            k_min = k_cat.min(dim=0).values
            k_max = k_cat.max(dim=0).values
            scale = (k_max - k_min) / 2.0
            zero = (k_max + k_min) / 2.0
            k_norm = torch.clamp((k_cat - zero) / (scale + 1e-8), -1, 1)
            k_dense, _, k_mask = extract_outliers(
                k_norm, outlier_ratio=outlier_ratio, dim=0
            )
            fisher = fisher_k[i].flatten()
            weights = fisher.unsqueeze(0).expand_as(k_dense)
            k_train = k_dense[~k_mask] if k_mask.any() else k_dense.reshape(-1)
            w_train = weights[~k_mask] if k_mask.any() else weights.reshape(-1)
            if k_train.numel() == 0:
                k_train, w_train = k_dense.reshape(-1), weights.reshape(-1)
            k_codebooks.append(train_nuq_codebook(k_train, w_train, n_bits=bit))
            k_scales.append(scale)
            k_zeros.append(zero)

            v_cat = torch.cat(v_acts[i], dim=0)
            v_min = v_cat.min(dim=-1, keepdim=True).values
            v_max = v_cat.max(dim=-1, keepdim=True).values
            v_scale = (v_max - v_min) / 2.0
            v_zero = (v_max + v_min) / 2.0
            v_norm = torch.clamp((v_cat - v_zero) / (v_scale + 1e-8), -1, 1)
            v_dense, _, v_mask = extract_outliers(
                v_norm, outlier_ratio=outlier_ratio, dim=-1
            )
            fisher_v_i = fisher_v[i].flatten()
            weights_v = fisher_v_i.unsqueeze(0).expand_as(v_dense)
            v_train = v_dense[~v_mask] if v_mask.any() else v_dense.reshape(-1)
            wv_train = weights_v[~v_mask] if v_mask.any() else weights_v.reshape(-1)
            if v_train.numel() == 0:
                v_train, wv_train = v_dense.reshape(-1), weights_v.reshape(-1)
            v_codebooks.append(train_nuq_codebook(v_train, wv_train, n_bits=bit))

        # 构造该比特宽度下的量化器
        quantizer = KVQuantizer(
            k_codebooks=k_codebooks,
            v_codebooks=v_codebooks,
            k_scales=k_scales,
            k_zeros=k_zeros,
            k_bits=bit,
            v_bits=bit,
            outlier_ratio=outlier_ratio,
            sink_token=sink_token,
            use_dense_sparse=outlier_ratio > 0,
        )

        # 对每层单独量化，测量损失增量
        from kvquant.attention_patch import patch_model, unpatch_model

        for layer_idx in range(n_layers):
            # 只量化当前层，其余层保持 fp16
            _patch_single_layer(model, quantizer, layer_idx)
            loss = _compute_loss(model, tokenizer, calib_texts, seq_len, device)
            sensitivity[layer_idx][bit] = abs(loss - baseline_loss)
            unpatch_model(model)

    return sensitivity


def _compute_loss(model, tokenizer, texts, seq_len, device):
    """在校准集上计算平均 loss。"""
    model.eval()
    total_loss = 0.0
    n_valid = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=seq_len
            ).to(device)
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
            n_valid += 1

    return total_loss / max(n_valid, 1)


def _patch_single_layer(model, quantizer, layer_idx):
    """只对指定层挂 hook。"""
    layer = model.model.layers[layer_idx]
    k_proj = layer.self_attn.k_proj
    v_proj = layer.self_attn.v_proj

    def make_k_hook(idx):
        def hook(module, inp, out):
            assert out.dim() == 3
            return quantizer.quantize_k(idx, out)
        return hook

    def make_v_hook(idx):
        def hook(module, inp, out):
            assert out.dim() == 3
            return quantizer.quantize_v(idx, out)
        return hook

    k_proj.register_forward_hook(make_k_hook(layer_idx))
    v_proj.register_forward_hook(make_v_hook(layer_idx))


def allocate_bits(
    sensitivity: dict,
    n_layers: int,
    candidate_bits: list = None,
    avg_bits: float = 3.0,
):
    """在全局平均比特预算下分配每层比特宽度（论文 §3.6）。

    策略：按敏感度分数排序，敏感度高的层分配更多比特，
    直到满足平均比特预算。

    返回:
        layer_bits: dict, {layer_idx: bit_width}
    """
    if candidate_bits is None:
        candidate_bits = [2, 3, 4]

    # 计算每层的综合敏感度（各比特宽度的平均损失增量）
    layer_scores = {}
    for i in range(n_layers):
        scores = [sensitivity[i].get(b, 0.0) for b in candidate_bits]
        layer_scores[i] = sum(scores) / len(scores)

    # 按敏感度从高到低排序
    sorted_layers = sorted(layer_scores.items(), key=lambda x: -x[1])

    # 贪心分配：从最低比特开始，敏感度高的层优先提升
    layer_bits = {i: candidate_bits[0] for i in range(n_layers)}
    total_bits = sum(layer_bits.values())

    # 按敏感度排序，依次尝试提升比特
    improved = True
    while improved:
        improved = False
        for i, _ in sorted_layers:
            current_bit = layer_bits[i]
            idx = candidate_bits.index(current_bit)
            if idx + 1 < len(candidate_bits):
                next_bit = candidate_bits[idx + 1]
                # 检查提升后平均比特是否仍在预算内
                new_total = total_bits - current_bit + next_bit
                new_avg = new_total / n_layers
                if new_avg <= avg_bits:
                    layer_bits[i] = next_bit
                    total_bits = new_total
                    improved = True

    return layer_bits