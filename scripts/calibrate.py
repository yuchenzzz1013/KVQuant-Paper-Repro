import argparse
import pickle

import torch
from datasets import load_dataset

from kvquant.model_loader import load_hf_model
from kvquant.calibration import calibrate
from kvquant.sensitivity import estimate_layer_sensitivity, allocate_bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model")
    parser.add_argument("--output", default="kvquant_calib.pkl")
    parser.add_argument("--n_bits", type=int, default=3,
                        help="默认比特宽度（当 --mixed_precision 关闭时使用）")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="启用逐层混合精度（论文 §3.6）")
    parser.add_argument("--avg_bits", type=float, default=3.0,
                        help="混合精度下的全局平均比特预算")
    parser.add_argument("--candidate_bits", type=str, default="2,3,4",
                        help="候选比特宽度，逗号分隔")
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--outlier_ratio", type=float, default=0.01)
    parser.add_argument("--no_sink", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_hf_model(
        args.model_path, device_map=None, torch_dtype=torch.float16
    )
    model.to(device)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if t.strip()][: args.n_samples]

    candidate_bits = [int(b) for b in args.candidate_bits.split(",")]

    layer_bits = None
    if args.mixed_precision:
        print("=" * 60)
        print("步骤 1/2: 逐层敏感度分析")
        print("=" * 60)
        sensitivity = estimate_layer_sensitivity(
            model, tokenizer, texts,
            seq_len=args.seq_len,
            device=device,
            candidate_bits=candidate_bits,
            outlier_ratio=args.outlier_ratio,
            sink_token=not args.no_sink,
        )

        print("\n敏感度分数（损失增量）:")
        for i in range(len(model.model.layers)):
            scores = {b: sensitivity[i].get(b, 0.0) for b in candidate_bits}
            score_str = "  ".join(f"{b}bit={v:.4f}" for b, v in scores.items())
            print(f"  Layer {i:2d}: {score_str}")

        layer_bits = allocate_bits(
            sensitivity,
            n_layers=len(model.model.layers),
            candidate_bits=candidate_bits,
            avg_bits=args.avg_bits,
        )
        print(f"\n比特分配结果（平均 {args.avg_bits:.2f}-bit）:")
        for i, b in layer_bits.items():
            print(f"  Layer {i:2d}: {b}-bit")

    print("\n" + "=" * 60)
    print("步骤 2/2: 校准")
    print("=" * 60)
    calib = calibrate(
        model, tokenizer, texts,
        n_bits=args.n_bits,
        layer_bits=layer_bits,
        seq_len=args.seq_len,
        device=device,
        outlier_ratio=args.outlier_ratio,
        sink_token=not args.no_sink,
    )

    with open(args.output, "wb") as f:
        pickle.dump(calib, f)
    print(f"\nSaved calibration to {args.output}")


if __name__ == "__main__":
    main()