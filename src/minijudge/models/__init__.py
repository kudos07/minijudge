"""Model loading for 4-bit inference and optional LoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class LoadedJudge:
    model: Any
    tokenizer: Any
    model_name: str
    adapter_path: str | None


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_judge(
    model_name: str,
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
    device_map: str | dict = "auto",
) -> LoadedJudge:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = _bnb_config()
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded LoRA adapter from {adapter_path}")

    model.eval()
    return LoadedJudge(
        model=model,
        tokenizer=tok,
        model_name=model_name,
        adapter_path=adapter_path,
    )
