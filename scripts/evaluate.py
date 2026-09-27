import argparse
import pickle
import torch

from kvquant.model_loader import load_hf_model
from kvquant.quant.kv_quantizer import KVQuantizer
from kvquant.attention_patch import patch_model
from benchmarks.perplexity import evaluate_perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model")
    parser.add_argument("--calib", default="kvquant_calib.pkl")
    parser.add_argument("--n_bits", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=2048)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_hf_model(
        args.model_path, device_map=None, torch_dtype=torch.float16
    )
    model.to(device)

    with open(args.calib, "rb") as f:
        calib = pickle.load(f)

    quantizer = KVQuantizer(
        k_codebooks=calib["k_codebooks"],
        v_codebooks=calib["v_codebooks"],
        k_scales=calib["k_scales"],
        k_zeros=calib["k_zeros"],
        k_bits=args.n_bits,
        v_bits=args.n_bits,
        outlier_ratio=0.01,
        sink_token=True,
    )

    patch_model(model, quantizer)

    ppl = evaluate_perplexity(model, tokenizer, seq_len=args.seq_len, device=device)
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()