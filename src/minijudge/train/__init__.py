"""QLoRA training via PEFT+TRL (default) or Unsloth (optional)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from minijudge.prompts import build_chat_messages
from minijudge.utils import ensure_dir, peak_vram_gb, read_jsonl, reset_peak_vram, set_seed, write_json


def _rows_to_hf_dataset(rows: list[dict], tokenizer, max_seq_length: int, max_examples: int | None):
    from datasets import Dataset

    if max_examples:
        rows = rows[:max_examples]

    def to_text(ex: dict) -> dict:
        messages = build_chat_messages(ex["question"], ex["response_a"], ex["response_b"])
        messages = messages + [{"role": "assistant", "content": ex["label"].strip().upper()}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = Dataset.from_list(rows).map(to_text, remove_columns=[c for c in rows[0].keys()])
    return ds


def _train_peft(cfg: dict[str, Any]) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    # TRL API differs across versions; support both response_template packing styles
    try:
        from trl import SFTConfig  # type: ignore

        use_sft_config = True
    except ImportError:
        use_sft_config = False

    model_name = cfg["model_name"]
    out_dir = ensure_dir(cfg["output_dir"])
    set_seed(int(cfg.get("seed", 42)))
    reset_peak_vram()
    t0 = time.perf_counter()

    # RTX 40-series: prefer bf16. Mixing fp16 GradScaler with bf16 grads crashes:
    # RuntimeError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16
    use_bf16 = bool(cfg.get("bf16", False))
    use_fp16 = bool(cfg.get("fp16", True))
    if cfg.get("auto_precision", True) and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            use_bf16, use_fp16 = True, False
        else:
            use_bf16, use_fp16 = False, True
    # Never enable both AMP modes
    if use_bf16:
        use_fp16 = False
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Training precision: bf16={use_bf16} fp16={use_fp16} compute_dtype={compute_dtype}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.config.pad_token_id = tok.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True))
    )
    if cfg.get("gradient_checkpointing", True):
        model.config.use_cache = False

    lora = LoraConfig(
        r=int(cfg.get("lora_rank", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_rows = read_jsonl(cfg["train_path"])
    val_rows = read_jsonl(cfg["val_path"]) if cfg.get("val_path") else []
    max_seq = int(cfg.get("max_seq_length", 768))
    max_train = cfg.get("max_train_examples")

    train_ds = _rows_to_hf_dataset(train_rows, tok, max_seq, max_train)
    eval_ds = _rows_to_hf_dataset(val_rows, tok, max_seq, min(200, len(val_rows)) or None) if val_rows else None

    common_args = dict(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        logging_steps=int(cfg.get("logging_steps", 20)),
        save_steps=int(cfg.get("save_steps", 200)),
        eval_steps=int(cfg.get("eval_steps", 100)) if eval_ds is not None else None,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        fp16=use_fp16,
        bf16=use_bf16,
        optim=cfg.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type="cosine",
        report_to=cfg.get("report_to", "none"),
        seed=int(cfg.get("seed", 42)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        max_grad_norm=1.0,
        load_best_model_at_end=False,
    )

    if use_sft_config:
        # TRL >=0.21 / 1.x renamed max_seq_length -> max_length on SFTConfig
        sft_kwargs = {
            **{k: v for k, v in common_args.items() if v is not None},
            "dataset_text_field": "text",
            "packing": False,
        }
        import inspect

        sft_params = inspect.signature(SFTConfig.__init__).parameters
        if "max_length" in sft_params:
            sft_kwargs["max_length"] = max_seq
        elif "max_seq_length" in sft_params:
            sft_kwargs["max_seq_length"] = max_seq

        sft_args = SFTConfig(**sft_kwargs)
        trainer = SFTTrainer(
            model=model,
            processing_class=tok,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=sft_args,
        )
    else:
        args = TrainingArguments(**{k: v for k, v in common_args.items() if v is not None})
        trainer = SFTTrainer(
            model=model,
            processing_class=tok,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=args,
        )

    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    elapsed = time.perf_counter() - t0
    metrics = {
        "backend": "peft",
        "model_name": model_name,
        "output_dir": str(out_dir),
        "train_runtime_sec": elapsed,
        "train_loss": float(train_result.training_loss) if hasattr(train_result, "training_loss") else None,
        "metrics": dict(train_result.metrics) if hasattr(train_result, "metrics") else {},
        "peak_vram_gb": peak_vram_gb(),
        "n_train": len(train_ds),
    }
    write_json(Path(out_dir) / "train_metrics.json", metrics)
    print("Training complete:", metrics)
    return metrics


def _train_unsloth(cfg: dict[str, Any]) -> dict[str, Any]:
    """Optional Unsloth path — requires requirements-unsloth.txt."""
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    model_name = cfg["model_name"]
    out_dir = ensure_dir(cfg["output_dir"])
    set_seed(int(cfg.get("seed", 42)))
    reset_peak_vram()
    t0 = time.perf_counter()
    max_seq = int(cfg.get("max_seq_length", 768))

    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg.get("lora_rank", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        target_modules=cfg.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    train_rows = read_jsonl(cfg["train_path"])
    val_rows = read_jsonl(cfg["val_path"]) if cfg.get("val_path") else []
    train_ds = _rows_to_hf_dataset(train_rows, tok, max_seq, cfg.get("max_train_examples"))
    eval_ds = _rows_to_hf_dataset(val_rows, tok, max_seq, min(200, len(val_rows)) or None) if val_rows else None

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        logging_steps=int(cfg.get("logging_steps", 20)),
        save_steps=int(cfg.get("save_steps", 200)),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=cfg.get("optim", "adamw_8bit"),
        seed=int(cfg.get("seed", 42)),
        report_to=cfg.get("report_to", "none"),
        max_length=max_seq,
        dataset_text_field="text",
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
    )
    result = trainer.train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    metrics = {
        "backend": "unsloth",
        "model_name": model_name,
        "output_dir": str(out_dir),
        "train_runtime_sec": time.perf_counter() - t0,
        "metrics": dict(result.metrics) if hasattr(result, "metrics") else {},
        "peak_vram_gb": peak_vram_gb(),
        "n_train": len(train_ds),
    }
    write_json(Path(out_dir) / "train_metrics.json", metrics)
    return metrics


def train_qlora(cfg: dict[str, Any]) -> dict[str, Any]:
    backend = (cfg.get("backend") or "peft").lower()
    if backend == "unsloth":
        try:
            return _train_unsloth(cfg)
        except ImportError:
            print("Unsloth not installed - falling back to PEFT backend.")
    return _train_peft(cfg)
