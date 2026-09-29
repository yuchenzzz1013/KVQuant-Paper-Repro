# KVQuant 复现

> ⚠️ 仅为个人轻量级复现

- 论文：https://arxiv.org/abs/2401.18079
- 官方代码：https://github.com/SqueezeAILab/KVQuant

## 目录

```
kvquant-repro/
├── model/
├── kvquant/
│ ├── init.py
│ ├── model_loader.py
│ ├── calibration.py
│ ├── kv_cache.py
│ ├── attention_patch.py
│ ├── quant/
│ │ ├── init.py
│ │ ├── nuq.py
│ │ ├── dense_sparse.py
│ │ └── kv_quantizer.py
│ └── kernels/
│ ├── init.py
│ ├── cuda_ops.py 
│ ├── csrc/
│ │ └── kvquant_ops.cu
│ ├── rope_fused.py
│ └── matvec_ds.py
├── benchmarks/
│ ├── init.py
│ └── perplexity.py
└── scripts/
├── calibrate.py
└── evaluate.py
```