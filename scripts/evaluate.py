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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_hf_model(
        args.model_path, device_map=None, torch_dtype=torch.float16
    )
    model.to(device)
    model.config.use_cache = False  # 避免 KV cache 干扰 PPL

    with open(args.calib, "rb") as f:
        calib = pickle.load(f)

    quantizer = KVQuantizer(
        k_codebooks=calib["k_codebooks"],
        v_codebooks=calib["v_codebooks"],
        k_scales=calib["k_scales"],
        k_zeros=calib["k_zeros"],
        k_bits=args.n_bits,
        v_bits=args.n_bits,
        outlier_ratio=args.outlier_ratio,
        sink_token=calib.get("sink_token", True),
        use_dense_sparse=args.outlier_ratio > 0,
    )

    patch_model(model, quantizer)

    # fp16 baseline（先 unpatch 再测）
    try:
        ppl_fp16 = evaluate_perplexity(
            model, tokenizer, seq_len=args.seq_len, device=device
        )
        print(f"[fp16 baseline] Perplexity: {ppl_fp16:.4f}")
    except Exception as e:
        print(f"fp16 baseline failed: {e}")

    ppl_q = evaluate_perplexity(
        model, tokenizer, seq_len=args.seq_len, device=device
    )
    print(f"[KVQuant {args.n_bits}-bit] Perplexity: {ppl_q:.4f}")

    unpatch_model(model)


if __name__ == "__main__":
    main()