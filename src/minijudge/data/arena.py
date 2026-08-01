"""Chatbot Arena preference pair preparation."""

from __future__ import annotations

import random
from typing import Any, Optional

from datasets import load_dataset

from minijudge.hf_auth import ensure_hf_login
from minijudge.utils import ensure_dir, write_jsonl


def _conversation_turns(conversation: list[dict] | None) -> int:
    if not conversation:
        return 0
    # Count user turns
    return sum(1 for m in conversation if m.get("role") == "user")


def _first_user_text(conversation: list[dict] | None) -> str:
    if not conversation:
        return ""
    for m in conversation:
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _assistant_text(conversation: list[dict] | None) -> str:
    if not conversation:
        return ""
    for m in conversation:
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def filter_arena_row(row: dict[str, Any], cfg: dict[str, Any]) -> dict | None:
    """Return a normalized preference example or None if filtered out."""
    # Language
    lang = (row.get("language") or "").strip()
    want = cfg.get("language", "English")
    if want and lang and lang.lower() != want.lower():
        # Some rows use "English"; others may be empty — keep empty only if no filter
        if lang.lower() not in {want.lower(), "en", "eng"}:
            return None

    winner = row.get("winner")
    if cfg.get("drop_ties", True) and winner in {None, "tie", "tie (bothbad)"}:
        return None
    if winner not in {"model_a", "model_b"}:
        return None

    conv_a = row.get("conversation_a") or []
    conv_b = row.get("conversation_b") or []

    if cfg.get("single_turn_only", True):
        if _conversation_turns(conv_a) != 1 or _conversation_turns(conv_b) != 1:
            return None

    question = _first_user_text(conv_a) or _first_user_text(conv_b)
    response_a = _assistant_text(conv_a)
    response_b = _assistant_text(conv_b)
    if not question or not response_a or not response_b:
        return None

    max_q = int(cfg.get("max_question_chars", 800))
    max_r = int(cfg.get("max_response_chars", 1800))
    if len(question) > max_q:
        return None
    if len(response_a) > max_r or len(response_b) > max_r:
        return None

    label = "A" if winner == "model_a" else "B"
    return {
        "id": str(row.get("question_id") or row.get("id") or ""),
        "source": "chatbot_arena",
        "question": question,
        "response_a": response_a,
        "response_b": response_b,
        "label": label,
        "model_a": row.get("model_a"),
        "model_b": row.get("model_b"),
    }


def prepare_arena(cfg: dict[str, Any], hf_token: Optional[str] = None) -> dict[str, int]:
    """Download, filter, split, and write Arena JSONL files."""
    raw_dir = ensure_dir(cfg.get("raw_dir", "data/raw"))
    out_dir = ensure_dir(cfg.get("output_dir", "data/processed"))
    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)

    token = ensure_hf_login(hf_token)
    dataset_id = cfg.get("dataset_id", "lmsys/chatbot_arena_conversations")
    print(f"Loading {dataset_id} (split=train) ...")
    try:
        # Same API as HF docs:
        #   ds = load_dataset("lmsys/chatbot_arena_conversations")
        ds = load_dataset(dataset_id, split="train", token=token)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {dataset_id}.\n"
            "Make sure you:\n"
            "  - created an HF token with read access\n"
            "  - accepted terms on the dataset page\n"
            "  - ran: huggingface-cli login   OR set $env:HF_TOKEN\n"
            f"Original error: {e}"
        ) from e

    kept: list[dict] = []
    for row in ds:
        ex = filter_arena_row(row, cfg)
        if ex is not None:
            if not ex["id"]:
                ex["id"] = f"arena_{len(kept)}"
            kept.append(ex)

    rng.shuffle(kept)
    print(f"Kept {len(kept)} filtered preference pairs")

    n_train = int(cfg.get("n_train", 3000))
    n_val = int(cfg.get("n_val", 400))
    n_test = int(cfg.get("n_test", 600))
    n_smoke = int(cfg.get("n_smoke_train", 500))
    n_week1 = int(cfg.get("n_week1", 1000))

    need = n_train + n_val + n_test
    if len(kept) < need:
        raise RuntimeError(
            f"Only {len(kept)} examples after filtering; need at least {need}. "
            "Relax max_response_chars or language filters."
        )

    train = kept[:n_train]
    val = kept[n_train : n_train + n_val]
    test = kept[n_train + n_val : n_train + n_val + n_test]
    smoke = train[:n_smoke]
    week1 = kept[:n_week1]

    counts = {
        "filtered_total": len(kept),
        "train": write_jsonl(out_dir / "arena_train.jsonl", train),
        "val": write_jsonl(out_dir / "arena_val.jsonl", val),
        "test": write_jsonl(out_dir / "arena_test.jsonl", test),
        "smoke_train": write_jsonl(out_dir / "arena_smoke_train.jsonl", smoke),
        "week1": write_jsonl(out_dir / "arena_week1.jsonl", week1),
    }
    # Touch raw marker
    (raw_dir / "arena_downloaded.ok").write_text("ok\n", encoding="utf-8")
    print("Wrote:", counts)
    return counts
