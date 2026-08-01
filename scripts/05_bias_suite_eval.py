#!/usr/bin/env python
"""Step 5b: Evaluate the controlled bias suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minijudge.eval import evaluate
from minijudge.utils import load_yaml, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Bias-suite evaluation")
    ap.add_argument("--config", default="configs/eval_bias.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    # Ensure we only hit the bias suite path
    cfg.pop("data_path", None)
    cfg.pop("judgebench_path", None)
    if not cfg.get("bias_suite_path"):
        cfg["bias_suite_path"] = "data/processed/bias_suite.jsonl"

    reports = {}
    # Fine-tuned (or whatever adapter_path points to)
    reports["with_config"] = evaluate(cfg)

    if cfg.get("compare_base"):
        base_cfg = dict(cfg)
        base_cfg["adapter_path"] = None
        base_cfg["experiment_name"] = (cfg.get("experiment_name") or "bias") + "_base"
        base_cfg["output_dir"] = str(Path(cfg["output_dir"]) / "base")
        reports["base"] = evaluate(base_cfg)

    write_json(Path(cfg["output_dir"]) / "bias_comparison.json", reports)
    print("Bias comparison written.")


if __name__ == "__main__":
    main()
