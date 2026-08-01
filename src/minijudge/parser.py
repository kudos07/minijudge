"""Strict A/B output parsing for tiny judges."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParseResult:
    label: str | None  # "A", "B", or None if invalid
    raw: str
    valid: bool


_LETTER = re.compile(
    r"(?:^|[^A-Za-z])([AB])(?:[^A-Za-z]|$)|"
    r"\b(?:response|option|choice|answer)\s*[:\-]?\s*([AB])\b|"
    r"\b([AB])\s*(?:is\s+better|wins?|is\s+preferable)\b",
    re.IGNORECASE,
)


def parse_ab_label(text: str) -> ParseResult:
    """Extract a single A/B decision from model output.

    Preference order:
    1. Exact "A" or "B" (optionally wrapped in punctuation)
    2. First clear letter match via patterns
    3. Invalid otherwise
    """
    raw = (text or "").strip()
    if not raw:
        return ParseResult(label=None, raw=raw, valid=False)

    # Exact / near-exact
    compact = raw.strip(" \t\n\r.\"'`*_()[]{}:")
    if compact.upper() in {"A", "B"}:
        return ParseResult(label=compact.upper(), raw=raw, valid=True)

    # First line only often holds the label for well-behaved models
    first = raw.splitlines()[0].strip().strip(" \t.\"'`*_()[]{}:")
    if first.upper() in {"A", "B"}:
        return ParseResult(label=first.upper(), raw=raw, valid=True)

    matches = []
    for m in _LETTER.finditer(raw):
        letter = next(g for g in m.groups() if g)
        matches.append(letter.upper())

    if not matches:
        return ParseResult(label=None, raw=raw, valid=False)

    # Ambiguous if both letters appear as candidates
    uniq = set(matches)
    if len(uniq) == 1:
        return ParseResult(label=matches[0], raw=raw, valid=True)

    # Prefer the first match if both appear (common: "A is better than B")
    return ParseResult(label=matches[0], raw=raw, valid=True)


def invert_label(label: str | None) -> str | None:
    """Map A<->B after position swap back to original identity."""
    if label is None:
        return None
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    return None
