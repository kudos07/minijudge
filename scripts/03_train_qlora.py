#!/usr/bin/env python
"""Step 3/4: QLoRA fine-tune Qwen3-1.7B as a pairwise judge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minijudge.train import train_qlora
from minijudge.utils import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Train MiniJudge with QLoRA")
    ap.add_argument("--config", default="configs/train_smoke.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    train_qlora(cfg)


if __name__ == "__main__":
    main()
