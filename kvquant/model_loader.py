from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_hf_model(
    model_path: Union[str, Path] = "model",
    device_map: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    attn_implementation: str = "eager",
    trust_remote_code: bool = False,
    local_files_only: bool = True,
    use_fast_tokenizer: bool = True,
    **kwargs,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    model_path = Path(model_path).expanduser().resolve()

    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"{model_path} 下没有 config.json，不是标准 HF 模型目录")

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=use_fast_tokenizer,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = dict(
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        **kwargs,
    )

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    model.eval()
    return model, tokenizer