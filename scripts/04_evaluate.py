#!/usr/bin/env python
"""Step 5: Evaluate fine-tuned / reliability configurations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minijudge.eval import evaluate
from minijudge.utils import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate MiniJudge model")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.max_examples is not None:
        cfg["max_examples"] = args.max_examples
    evaluate(cfg)


if __name__ == "__main__":
    main()
