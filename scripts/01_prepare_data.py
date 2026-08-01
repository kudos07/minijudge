#!/usr/bin/env python
"""Step 1: Prepare Chatbot Arena + JudgeBench + bias suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minijudge.data.arena import prepare_arena
from minijudge.data.bias_suite import prepare_bias_suite
from minijudge.data.judgebench import prepare_judgebench
from minijudge.hf_auth import ensure_hf_login
from minijudge.utils import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare MiniJudge datasets")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (else uses HF_TOKEN env or cached login)",
    )
    ap.add_argument(
        "--skip-arena",
        action="store_true",
        help="Only build JudgeBench + bias suite (no gated Arena download)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    if not args.skip_arena:
        print("=== Hugging Face auth ===")
        ensure_hf_login(args.token)
        print("=== Chatbot Arena ===")
        prepare_arena(cfg, hf_token=args.token)
    else:
        print("Skipping Chatbot Arena download (--skip-arena).")

    print("=== JudgeBench ===")
    prepare_judgebench(cfg)
    print("=== Bias suite ===")
    prepare_bias_suite(cfg.get("output_dir", "data/processed"))
    print("Done. Files are under data/processed/")


if __name__ == "__main__":
    main()
