"""Evaluation package. Heavy torch imports live in judge.py."""

from __future__ import annotations

from typing import Any


def evaluate(cfg: dict[str, Any]):
    from minijudge.eval.judge import evaluate as _evaluate

    return _evaluate(cfg)


__all__ = ["evaluate"]
