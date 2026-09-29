import argparse
import pickle
import torch

from kvquant.model_loader import load_hf_model
from kvquant.quant.kv_quantizer import KVQuantizer
from kvquant.attention_patch import patch_model, unpatch_model
from benchmarks.perplexity import evaluate_perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model")
    parser.add_argument("--calib", default="kvquant_calib.pkl")
    parser.add_argument("--n_bits", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--outlier_ratio", type=float, default=0.01)
    parser.add_argument("--skip_fp16", action="store_true",
                        help="跳过 fp16 baseline（节省时间）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_hf_model(
        args.model_path, device_map=None, torch_dtype=torch.float16
    )
    model.to(device)
    model.config.use_cache = False  # 避免 KV cache 干扰 PPL

    # ============ 1. fp16 baseline（必须在 patch 之前测）============
    if not args.skip_fp16:
        ppl_fp16 = evaluate_perplexity(
            model, tokenizer, seq_len=args.seq_len, device=device
        )
        print(f"[fp16 baseline] Perplexity: {ppl_fp16:.4f}")

    # ============ 2. 加载校准结果（以文件内记录为准）============
    with open(args.calib, "rb") as f:
        calib = pickle.load(f)

    n_bits = calib.get("n_bits", args.n_bits)
    outlier_ratio = calib.get("outlier_ratio", args.outlier_ratio)
    sink_token = calib.get("sink_token", True)

    quantizer = KVQuantizer(
        k_codebooks=calib["k_codebooks"],
        v_codebooks=calib["v_codebooks"],
        k_scales=calib["k_scales"],
        k_zeros=calib["k_zeros"],
        k_bits=n_bits,
        v_bits=n_bits,
        outlier_ratio=outlier_ratio,
        sink_token=sink_token,
        use_dense_sparse=outlier_ratio > 0,
    )

    # ============ 3. patch 后测量化 PPL ============
    patch_model(model, quantizer)
    ppl_q = evaluate_perplexity(
        model, tokenizer, seq_len=args.seq_len, device=device
    )
    print(f"[KVQuant {n_bits}-bit] Perplexity: {ppl_q:.4f}")

    unpatch_model(model)


if __name__ == "__main__":
    main()