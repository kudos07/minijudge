#!/usr/bin/env python
"""Verify Hugging Face access to lmsys/chatbot_arena_conversations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minijudge.hf_auth import ensure_hf_login


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None)
    ap.add_argument("--dataset", default="lmsys/chatbot_arena_conversations")
    args = ap.parse_args()

    token = ensure_hf_login(args.token)
    from datasets import load_dataset

    print(f"Loading one stream sample from {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train", streaming=True, token=token)
    row = next(iter(ds))
    keys = sorted(row.keys())
    print("OK - dataset accessible.")
    print("Columns:", keys)
    print("winner sample:", row.get("winner"))
    print("language sample:", row.get("language"))
    print("model_a / model_b:", row.get("model_a"), "/", row.get("model_b"))


if __name__ == "__main__":
    main()
