"""Pairwise judge inference with optional position swap + majority voting."""

from __future__ import annotations

import time
from typing import Any

import torch
from tqdm import tqdm

from minijudge.eval.metrics import (
    compute_classification_metrics,
    majority_vote,
    position_consistency_metrics,
)
from minijudge.models import LoadedJudge, load_judge
from minijudge.parser import invert_label, parse_ab_label
from minijudge.prompts import build_chat_messages
from minijudge.utils import ensure_dir, peak_vram_gb, read_jsonl, reset_peak_vram, write_json, write_jsonl


@torch.inference_mode()
def generate_label(
    judge: LoadedJudge,
    question: str,
    response_a: str,
    response_b: str,
    *,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> tuple[str | None, str, float]:
    """Return (parsed_label, raw_text, latency_sec)."""
    tok = judge.tokenizer
    messages = build_chat_messages(question, response_a, response_b)
    try:
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: skip thinking for A/B labels
        )
    except TypeError:
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tok(prompt, return_tensors="pt")
    try:
        device = next(judge.model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if do_sample and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    t0 = time.perf_counter()
    out = judge.model.generate(**inputs, **gen_kwargs)
    latency = time.perf_counter() - t0

    new_tokens = out[0][inputs["input_ids"].shape[-1] :]
    raw = tok.decode(new_tokens, skip_special_tokens=True)
    parsed = parse_ab_label(raw)
    return parsed.label, raw, latency


def judge_example(
    judge: LoadedJudge,
    ex: dict,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run one example with optional swap + majority voting."""
    votes = int(cfg.get("majority_votes", 1))
    do_swap = bool(cfg.get("position_swap", False))
    temp = float(cfg.get("majority_temperature", cfg.get("temperature", 0.0)))
    do_sample = bool(cfg.get("do_sample", temp > 0))
    max_new = int(cfg.get("max_new_tokens", 8))

    # If majority_votes>1, use majority_temperature; else use temperature
    if votes > 1:
        run_temp = float(cfg.get("majority_temperature", 0.2))
        run_sample = run_temp > 0
    else:
        run_temp = float(cfg.get("temperature", 0.0))
        run_sample = bool(cfg.get("do_sample", False))

    latencies: list[float] = []
    ab_votes: list[str | None] = []
    ba_votes_mapped: list[str | None] = []
    raw_logs: list[dict] = []

    for _ in range(votes):
        lab, raw, lat = generate_label(
            judge,
            ex["question"],
            ex["response_a"],
            ex["response_b"],
            max_new_tokens=max_new,
            temperature=run_temp,
            do_sample=run_sample,
        )
        ab_votes.append(lab)
        latencies.append(lat)
        raw_logs.append({"order": "AB", "raw": raw, "label": lab})

    pred_ab = majority_vote(ab_votes) if votes > 1 else ab_votes[0]

    pred_ba_mapped = None
    if do_swap:
        for _ in range(votes):
            lab, raw, lat = generate_label(
                judge,
                ex["question"],
                ex["response_b"],  # swapped
                ex["response_a"],
                max_new_tokens=max_new,
                temperature=run_temp,
                do_sample=run_sample,
            )
            mapped = invert_label(lab)
            ba_votes_mapped.append(mapped)
            latencies.append(lat)
            raw_logs.append({"order": "BA", "raw": raw, "label": lab, "mapped": mapped})
        pred_ba_mapped = majority_vote(ba_votes_mapped) if votes > 1 else ba_votes_mapped[0]

    # Final prediction: if swap enabled and both available, require agreement or use AB
    if do_swap and pred_ab in {"A", "B"} and pred_ba_mapped in {"A", "B"}:
        final = pred_ab if pred_ab == pred_ba_mapped else pred_ab  # report AB; track conflict
    else:
        final = pred_ab

    return {
        "id": ex.get("id"),
        "label": ex["label"],
        "pred": final,
        "pred_ab": pred_ab,
        "pred_ba_mapped": pred_ba_mapped,
        "consistent": (
            pred_ab == pred_ba_mapped
            if pred_ab in {"A", "B"} and pred_ba_mapped in {"A", "B"}
            else None
        ),
        "latency_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_total": sum(latencies),
        "n_generations": len(latencies),
        "raw_logs": raw_logs,
        "bias_type": ex.get("bias_type"),
        "source": ex.get("source"),
        "category": ex.get("category"),
    }


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    reset_peak_vram()
    out_dir = ensure_dir(cfg["output_dir"])

    judge = load_judge(
        model_name=cfg["model_name"],
        adapter_path=cfg.get("adapter_path"),
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        device_map=cfg.get("device_map", "auto"),
    )

    data_paths = []
    if cfg.get("data_path"):
        data_paths.append(("arena_test", cfg["data_path"]))
    if cfg.get("judgebench_path"):
        data_paths.append(("judgebench", cfg["judgebench_path"]))
    if cfg.get("bias_suite_path"):
        data_paths.append(("bias_suite", cfg["bias_suite_path"]))

    all_reports: dict[str, Any] = {
        "experiment_name": cfg.get("experiment_name"),
        "model_name": cfg["model_name"],
        "adapter_path": cfg.get("adapter_path"),
        "config": {
            "position_swap": cfg.get("position_swap"),
            "majority_votes": cfg.get("majority_votes"),
            "temperature": cfg.get("temperature"),
        },
        "datasets": {},
    }

    for split_name, path in data_paths:
        rows = read_jsonl(path)
        max_ex = cfg.get("max_examples")
        if max_ex:
            rows = rows[: int(max_ex)]
        if not rows:
            print(f"Skip empty split {split_name}: {path}")
            continue

        print(f"Evaluating {split_name}: {len(rows)} examples")
        results = []
        for ex in tqdm(rows, desc=split_name):
            results.append(judge_example(judge, ex, cfg))

        y_true = [r["label"] for r in results]
        y_pred = [r["pred"] for r in results]
        cls = compute_classification_metrics(y_true, y_pred)

        pred_ab = [r["pred_ab"] for r in results]
        pred_ba = [r["pred_ba_mapped"] for r in results]
        if cfg.get("position_swap"):
            pos = position_consistency_metrics(pred_ab, pred_ba)
        else:
            pos = {}

        latencies = [r["latency_mean"] for r in results]
        report = {
            **cls,
            **pos,
            "latency_mean_sec": sum(latencies) / len(latencies) if latencies else 0.0,
            "latency_total_sec": sum(r["latency_total"] for r in results),
            "peak_vram_gb": peak_vram_gb(),
        }

        # Per bias-type breakdown
        if any(r.get("bias_type") for r in results):
            by_type: dict[str, dict] = {}
            types = sorted({r["bias_type"] for r in results if r.get("bias_type")})
            for bt in types:
                sub = [r for r in results if r.get("bias_type") == bt]
                by_type[bt] = compute_classification_metrics(
                    [r["label"] for r in sub],
                    [r["pred"] for r in sub],
                )
            report["by_bias_type"] = by_type

        write_jsonl(out_dir / f"{split_name}_predictions.jsonl", results)
        write_json(out_dir / f"{split_name}_metrics.json", report)
        all_reports["datasets"][split_name] = report
        print(f"{split_name} metrics:", report)

    all_reports["peak_vram_gb"] = peak_vram_gb()
    write_json(out_dir / "summary.json", all_reports)
    return all_reports
