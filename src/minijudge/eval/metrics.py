"""Evaluation metrics for pairwise LLM judges."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str | None],
) -> dict[str, Any]:
    """Accuracy / macro-F1 over valid predictions; track invalid rate separately."""
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    invalid = sum(1 for p in y_pred if p not in {"A", "B"})
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p in {"A", "B"}]

    if not valid_pairs:
        return {
            "n": n,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "invalid_output_rate": 1.0 if n else 0.0,
            "n_valid": 0,
            "confusion": None,
        }

    yt = [t for t, _ in valid_pairs]
    yp = [p for _, p in valid_pairs]
    cm = confusion_matrix(yt, yp, labels=["A", "B"]).tolist()
    return {
        "n": n,
        "n_valid": len(valid_pairs),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=["A", "B"], average="macro", zero_division=0)),
        "invalid_output_rate": invalid / n if n else 0.0,
        "confusion": {"labels": ["A", "B"], "matrix": cm},
        "pred_distribution": dict(Counter(yp)),
        "label_distribution": dict(Counter(yt)),
    }


def position_consistency_metrics(
    pred_ab: list[str | None],
    pred_ba_mapped: list[str | None],
) -> dict[str, Any]:
    """Consistency between normal and position-swapped (mapped-back) predictions."""
    assert len(pred_ab) == len(pred_ba_mapped)
    consistent = 0
    conflict = 0
    comparable = 0
    for a, b in zip(pred_ab, pred_ba_mapped):
        if a not in {"A", "B"} or b not in {"A", "B"}:
            continue
        comparable += 1
        if a == b:
            consistent += 1
        else:
            conflict += 1
    return {
        "n_comparable": comparable,
        "position_consistency": consistent / comparable if comparable else 0.0,
        "conflict_rate": conflict / comparable if comparable else 0.0,
    }


def majority_vote(labels: list[str | None]) -> str | None:
    valid = [x for x in labels if x in {"A", "B"}]
    if not valid:
        return None
    counts = Counter(valid)
    # Stable tie-break: prefer A
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best
