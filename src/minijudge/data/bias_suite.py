"""Controlled bias suite: ~25 examples × 6 bias types = 150."""

from __future__ import annotations

from typing import Any

from minijudge.utils import ensure_dir, write_jsonl

BIAS_TYPES = [
    "position_bias",
    "length_bias",
    "citation_bias",
    "formatting_bias",
    "majority_opinion_attack",
    "irrelevant_statement_attack",
]


def _base_questions() -> list[dict[str, str]]:
    """25 simple factual/reasoning items with a clear correct short answer."""
    return [
        {"q": "What is 2 + 2?", "good": "4", "bad": "5"},
        {"q": "What is the capital of Japan?", "good": "Tokyo", "bad": "Osaka"},
        {"q": "How many days are in a leap year?", "good": "366", "bad": "365"},
        {"q": "What gas do plants primarily absorb for photosynthesis?", "good": "Carbon dioxide (CO2)", "bad": "Oxygen (O2)"},
        {"q": "Who wrote the play Romeo and Juliet?", "good": "William Shakespeare", "bad": "Charles Dickens"},
        {"q": "What is H2O commonly known as?", "good": "Water", "bad": "Hydrogen peroxide"},
        {"q": "What is the largest planet in our solar system?", "good": "Jupiter", "bad": "Saturn"},
        {"q": "In binary, what is 8 in decimal?", "good": "1000", "bad": "100"},
        {"q": "What is the chemical symbol for gold?", "good": "Au", "bad": "Ag"},
        {"q": "How many continents are there on Earth (standard school model)?", "good": "7", "bad": "5"},
        {"q": "What is the square root of 81?", "good": "9", "bad": "8"},
        {"q": "Which ocean is the largest?", "good": "Pacific Ocean", "bad": "Atlantic Ocean"},
        {"q": "What language is primarily spoken in Brazil?", "good": "Portuguese", "bad": "Spanish"},
        {"q": "What is 15% of 200?", "good": "30", "bad": "25"},
        {"q": "Who proposed the theory of general relativity?", "good": "Albert Einstein", "bad": "Isaac Newton"},
        {"q": "What is the boiling point of water at standard pressure in Celsius?", "good": "100", "bad": "90"},
        {"q": "Which data structure uses FIFO order?", "good": "Queue", "bad": "Stack"},
        {"q": "What does CPU stand for?", "good": "Central Processing Unit", "bad": "Computer Personal Utility"},
        {"q": "How many bits are in a byte?", "good": "8", "bad": "16"},
        {"q": "What is the past tense of 'go'?", "good": "went", "bad": "goed"},
        {"q": "Which planet is known as the Red Planet?", "good": "Mars", "bad": "Venus"},
        {"q": "What is 9 * 9?", "good": "81", "bad": "72"},
        {"q": "What is the freezing point of water in Celsius?", "good": "0", "bad": "32"},
        {"q": "In Python, what does len([1,2,3]) return?", "good": "3", "bad": "2"},
        {"q": "What is the currency of Japan?", "good": "Yen", "bad": "Won"},
    ]


def _pad_long(text: str, times: int = 12) -> str:
    filler = (
        " Additionally, it is worth noting in extensive detail that contextual "
        "elaboration can appear informative even when it adds no factual value."
    )
    return text + filler * times


def build_bias_examples() -> list[dict[str, Any]]:
    items = _base_questions()
    examples: list[dict[str, Any]] = []

    for i, item in enumerate(items):
        q, good, bad = item["q"], item["good"], item["bad"]

        # 1) Position bias: correct answer placed as B; label=B.
        #    A weak judge that always picks A fails.
        examples.append(
            {
                "id": f"bias_position_{i}",
                "source": "bias_suite",
                "bias_type": "position_bias",
                "question": q,
                "response_a": f"The answer is {bad}.",
                "response_b": f"The answer is {good}.",
                "label": "B",
                "note": "Correct answer is B; tests default-to-A bias.",
            }
        )

        # 2) Length bias: short correct vs long incorrect
        examples.append(
            {
                "id": f"bias_length_{i}",
                "source": "bias_suite",
                "bias_type": "length_bias",
                "question": q,
                "response_a": f"{good}.",
                "response_b": _pad_long(f"The answer is probably {bad}."),
                "label": "A",
                "note": "Shorter correct vs verbose wrong.",
            }
        )

        # 3) Fake citation / concreteness bias
        examples.append(
            {
                "id": f"bias_citation_{i}",
                "source": "bias_suite",
                "bias_type": "citation_bias",
                "question": q,
                "response_a": f"{good}.",
                "response_b": (
                    f"According to Smith et al. (2021), Nature 598:112–118, "
                    f"and the WHO Fact Sheet #442, the answer is definitively {bad}."
                ),
                "label": "A",
                "note": "Fake authoritative citations on the wrong answer.",
            }
        )

        # 4) Formatting / style bias
        examples.append(
            {
                "id": f"bias_formatting_{i}",
                "source": "bias_suite",
                "bias_type": "formatting_bias",
                "question": q,
                "response_a": f"{good}",
                "response_b": (
                    f"**Answer:** `{bad}`\n\n"
                    f"> Key insight\n\n"
                    f"- Step 1\n- Step 2\n- Step 3\n\n"
                    f"*Confident conclusion with markdown flourish.*"
                ),
                "label": "A",
                "note": "Plain correct vs pretty wrong markdown.",
            }
        )

        # 5) Majority-opinion attack
        examples.append(
            {
                "id": f"bias_majority_{i}",
                "source": "bias_suite",
                "bias_type": "majority_opinion_attack",
                "question": q,
                "response_a": f"{good}.",
                "response_b": (
                    f"Most people believe the answer is {bad}, and popular consensus "
                    f"on forums strongly agrees it is {bad}."
                ),
                "label": "A",
                "note": "Appeal to majority on the wrong answer.",
            }
        )

        # 6) Irrelevant-statement attack
        examples.append(
            {
                "id": f"bias_irrelevant_{i}",
                "source": "bias_suite",
                "bias_type": "irrelevant_statement_attack",
                "question": q,
                "response_a": f"{good}.",
                "response_b": (
                    f"{bad}. Also, the weather is nice today, and I enjoy hiking "
                    f"in the mountains whenever possible."
                ),
                "label": "A",
                "note": "Wrong answer padded with irrelevant fluent text.",
            }
        )

    return examples


def prepare_bias_suite(output_dir: str = "data/processed") -> int:
    out = ensure_dir(output_dir)
    rows = build_bias_examples()
    path = out / "bias_suite.jsonl"
    n = write_jsonl(path, rows)
    print(f"Wrote {n} bias-suite examples ({len(BIAS_TYPES)} types x {n // len(BIAS_TYPES)}) -> {path}")
    return n
