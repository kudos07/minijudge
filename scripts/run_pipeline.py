#!/usr/bin/env python
"""Orchestrate MiniJudge stages: data → baseline → smoke → train → eval → bias."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGES = {
    "data": [["python", "scripts/01_prepare_data.py", "--config", "configs/data.yaml"]],
    "baseline": [
        ["python", "scripts/02_run_baseline.py", "--config", "configs/baseline_0.6b.yaml"],
        ["python", "scripts/02_run_baseline.py", "--config", "configs/baseline_1.7b.yaml"],
    ],
    "smoke": [["python", "scripts/03_train_qlora.py", "--config", "configs/train_smoke.yaml"]],
    "train": [["python", "scripts/03_train_qlora.py", "--config", "configs/train_qlora.yaml"]],
    "eval": [
        ["python", "scripts/04_evaluate.py", "--config", "configs/eval_qlora.yaml"],
        ["python", "scripts/04_evaluate.py", "--config", "configs/eval_qlora_reliability.yaml"],
    ],
    "bias": [["python", "scripts/05_bias_suite_eval.py", "--config", "configs/eval_bias.yaml"]],
}

ORDER = ["data", "baseline", "smoke", "train", "eval", "bias"]


def run_stage(name: str) -> None:
    print(f"\n======== STAGE: {name} ========\n")
    for cmd in STAGES[name]:
        print("+", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        default="data",
        help="One of: data, baseline, smoke, train, eval, bias, all",
    )
    args = ap.parse_args()

    if args.stage == "all":
        for s in ORDER:
            run_stage(s)
    elif args.stage in STAGES:
        run_stage(args.stage)
    else:
        ap.error(f"Unknown stage {args.stage!r}. Choose from {ORDER + ['all']}")


if __name__ == "__main__":
    main()
