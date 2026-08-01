"""JudgeBench loader — held-out objective correctness challenge set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minijudge.utils import ensure_dir, read_jsonl, write_jsonl


def _normalize_judgebench_row(row: dict[str, Any], idx: int) -> dict | None:
    """Map common JudgeBench / pairwise schemas into our A/B format."""
    question = (
        row.get("question")
        or row.get("prompt")
        or row.get("instruction")
        or row.get("input")
        or ""
    ).strip()
    response_a = (
        row.get("response_a")
        or row.get("answer_a")
        or row.get("output_a")
        or row.get("response1")
        or ""
    ).strip()
    response_b = (
        row.get("response_b")
        or row.get("answer_b")
        or row.get("output_b")
        or row.get("response2")
        or ""
    ).strip()

    label_raw = (
        row.get("label")
        or row.get("winner")
        or row.get("better")
        or row.get("correct_label")
        or ""
    )
    label = str(label_raw).strip().upper()
    # Map common encodings
    mapping = {
        "A": "A",
        "B": "B",
        "MODEL_A": "A",
        "MODEL_B": "B",
        "RESPONSE_A": "A",
        "RESPONSE_B": "B",
        "1": "A",
        "2": "B",
        "0": "A",
    }
    label = mapping.get(label, label)
    if label not in {"A", "B"}:
        return None
    if not question or not response_a or not response_b:
        return None

    return {
        "id": str(row.get("id") or f"judgebench_{idx}"),
        "source": "judgebench",
        "question": question,
        "response_a": response_a,
        "response_b": response_b,
        "label": label,
        "category": row.get("category") or row.get("split") or row.get("domain"),
    }


def _try_hf(dataset_id: str, max_examples: int | None) -> list[dict]:
    from datasets import load_dataset

    # Try a few common split names
    last_err = None
    for split in ("test", "validation", "train", "default"):
        try:
            ds = load_dataset(dataset_id, split=split)
            rows = []
            for i, row in enumerate(ds):
                ex = _normalize_judgebench_row(dict(row), i)
                if ex:
                    rows.append(ex)
                if max_examples and len(rows) >= max_examples:
                    break
            if rows:
                return rows
        except Exception as e:  # noqa: BLE001 — try next split
            last_err = e
            continue
    if last_err:
        print(f"JudgeBench HF load note: {last_err}")
    return []


def _builtin_seed_examples() -> list[dict]:
    """Tiny offline seed so the pipeline works before JudgeBench download."""
    return [
        {
            "id": "jb_seed_math_1",
            "source": "judgebench_seed",
            "question": "What is 17 * 19?",
            "response_a": "17 * 19 = 323.",
            "response_b": "17 * 19 = 333.",
            "label": "A",
            "category": "math",
        },
        {
            "id": "jb_seed_code_1",
            "source": "judgebench_seed",
            "question": "Write a Python function that returns the length of a list.",
            "response_a": "def length(xs):\n    return len(xs)",
            "response_b": "def length(xs):\n    return xs + 1",
            "label": "A",
            "category": "coding",
        },
        {
            "id": "jb_seed_know_1",
            "source": "judgebench_seed",
            "question": "What is the capital of France?",
            "response_a": "Lyon is the capital of France.",
            "response_b": "Paris is the capital of France.",
            "label": "B",
            "category": "knowledge",
        },
        {
            "id": "jb_seed_reason_1",
            "source": "judgebench_seed",
            "question": "If all bloops are razzies and all razzies are laddies, are all bloops laddies?",
            "response_a": "Yes. By transitivity of the stated inclusions, all bloops are laddies.",
            "response_b": "No. Being a razzie does not imply being a laddie for bloops.",
            "label": "A",
            "category": "reasoning",
        },
    ]


def prepare_judgebench(cfg: dict[str, Any]) -> int:
    """Write data/processed/judgebench.jsonl."""
    jb = cfg.get("judgebench") or {}
    if not jb.get("enabled", True):
        return 0

    out_dir = ensure_dir(cfg.get("output_dir", "data/processed"))
    out_path = out_dir / "judgebench.jsonl"
    max_examples = jb.get("max_examples")
    rows: list[dict] = []

    hf_id = jb.get("hf_dataset")
    if hf_id:
        print(f"Trying JudgeBench from HF: {hf_id}")
        rows = _try_hf(hf_id, max_examples)

    fallback = Path(jb.get("local_fallback", "data/raw/judgebench_sample.jsonl"))
    if not rows and fallback.exists():
        print(f"Using local JudgeBench fallback: {fallback}")
        for i, row in enumerate(read_jsonl(fallback)):
            ex = _normalize_judgebench_row(row, i)
            if ex:
                rows.append(ex)

    if not rows:
        print("JudgeBench unavailable - writing builtin seed examples for pipeline testing.")
        rows = _builtin_seed_examples()
        ensure_dir(fallback.parent)
        write_jsonl(fallback, rows)

    if max_examples:
        rows = rows[: int(max_examples)]

    n = write_jsonl(out_path, rows)
    print(f"Wrote {n} JudgeBench examples -> {out_path}")
    return n
