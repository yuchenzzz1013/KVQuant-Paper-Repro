# KVQuant 复现（个人轻量级 PyTorch 版）

> ⚠️ 仅为个人轻量级复现

- 论文：https://arxiv.org/abs/2401.18079
- 官方代码：https://github.com/SqueezeAILab/KVQuant

## 目录

```
kvquant-repro/
├── model/                        
├── kvquant/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── calibration.py
│   ├── kv_cache.py
│   ├── attention_patch.py
│   ├── quant/
│   │   ├── __init__.py
│   │   ├── nuq.py
│   │   ├── dense_sparse.py
│   │   └── kv_quantizer.py
│   └── kernels/
│       ├── __init__.py
│       ├── rope_fused.py
│       └── matvec_ds.py
├── benchmarks/
│   ├── __init__.py
│   └── perplexity.py
└── scripts/
    ├── calibrate.py
    └── evaluate.py
```