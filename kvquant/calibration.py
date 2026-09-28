import torch
from tqdm import tqdm

from kvquant.quant.nuq import train_nuq_codebook
from kvquant.quant.dense_sparse import extract_outliers


def calibrate(
    model,
    tokenizer,
    calib_texts,
    n_bits: int = 3,
    seq_len: int = 2048,
    device: str = "cuda",
    outlier_ratio: float = 0.01,
    sink_token: bool = True,
):
    """离线校准（论文 §3.3 / §3.4 / §3.5 / §3.6）。

    - Key:   Per-Channel Pre-RoPE
             -> 逐通道 scale/zero（离线）
             -> 逐层 NUQ codebook（Fisher 加权 KMeans）
    - Value: Per-Token
             -> 逐层 NUQ codebook（Fisher 加权 KMeans）
             -> scale/zero 留给在线逐 token 计算
    """
    model.eval()
    n_layers = len(model.model.layers)

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

    fisher_k = {i: None for i in range(n_layers)}
    fisher_v = {i: None for i in range(n_layers)}
    k_acts = {i: [] for i in range(n_layers)}
    v_acts = {i: [] for i in range(n_layers)}

    for text in tqdm(calib_texts, desc="Calibrating"):
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
                g = (k_grad.float() ** 2).sum(dim=(0, 1)).cpu()  # [H]
                fisher_k[i] = g if fisher_k[i] is None else fisher_k[i] + g

            if v_act is not None and v_grad is not None:
                if sink_token and v_act.size(1) > 1:
                    v_act = v_act[:, 1:, :]
                    v_grad = v_grad[:, 1:, :]
                v_acts[i].append(v_act.float().cpu())
                g = (v_grad.float() ** 2).sum(dim=(0, 1)).cpu()  # [H]
                fisher_v[i] = g if fisher_v[i] is None else fisher_v[i] + g

        act_store.clear()
        grad_store.clear()

    for h in hooks:
        h.remove()

    k_codebooks, v_codebooks = [], []
    k_scales, k_zeros = [], []

    for i in range(n_layers):
        # ---------- Key: Per-Channel Pre-RoPE ----------
        k_cat = torch.cat(k_acts[i], dim=0)                 # [N*S, H]
        k_min = k_cat.min(dim=0).values                     # [H]
        k_max = k_cat.max(dim=0).values
        scale = (k_max - k_min) / 2.0
        zero = (k_max + k_min) / 2.0

        k_norm = torch.clamp((k_cat - zero) / (scale + 1e-8), -1, 1)
        # dim=0 沿 token 维剔除 outlier（per-channel）
        k_dense, _, k_mask = extract_outliers(
            k_norm, outlier_ratio=outlier_ratio, dim=0
        )

        fisher = fisher_k[i].flatten()                       # [H]
        weights = fisher.unsqueeze(0).expand_as(k_dense)     # [N*S, H]

        # ★ 关键修复：只用非 outlier 元素训练码本
        k_train = k_dense[~k_mask]
        w_train = weights[~k_mask]

        if k_train.numel() == 0:
            # 极端情况：全被判为 outlier，退化为对全部元素训练
            k_train = k_dense.reshape(-1)
            w_train = weights.reshape(-1)

        k_codebooks.append(train_nuq_codebook(k_train, w_train, n_bits=n_bits))
        k_scales.append(scale)
        k_zeros.append(zero)

        # ---------- Value: Per-Token ----------
        v_cat = torch.cat(v_acts[i], dim=0)                 # [N*S, H]
        v_min = v_cat.min(dim=-1, keepdim=True).values
        v_max = v_cat.max(dim=-1, keepdim=True).values
        v_scale = (v_max - v_min) / 2.0
        v_zero = (v_max + v_min) / 2.0

        v_norm = torch.clamp((v_cat - v_zero) / (v_scale + 1e-8), -1, 1)
        # dim=-1 沿 channel 维剔除 outlier（per-token）
        v_dense, _, v_mask = extract_outliers(
            v_norm, outlier_ratio=outlier_ratio, dim=-1
        )

        fisher_v_i = fisher_v[i].flatten()                    # [H]
        weights_v = fisher_v_i.unsqueeze(0).expand_as(v_dense)

        v_train = v_dense[~v_mask]
        wv_train = weights_v[~v_mask]

        if v_train.numel() == 0:
            v_train = v_dense.reshape(-1)
            wv_train = weights_v.reshape(-1)

        v_codebooks.append(train_nuq_codebook(v_train, wv_train, n_bits=n_bits))

    return {
        "k_codebooks": k_codebooks,
        "v_codebooks": v_codebooks,
        "k_scales": k_scales,
        "k_zeros": k_zeros,
        "outlier_ratio": outlier_ratio,
        "sink_token": sink_token,
        "n_bits": n_bits,
    }