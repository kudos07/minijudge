"""Judge prompt templates — A/B label only (no rationales in v1)."""

JUDGE_SYSTEM = (
    "You are an impartial evaluator. "
    "Return exactly one label: A or B."
)

JUDGE_USER_TEMPLATE = """You are an impartial evaluator.

Evaluate which response better answers the user's question.

Consider:
1. Correctness
2. Relevance
3. Clarity
4. Completeness

Question:
{question}

Response A:
{response_a}

Response B:
{response_b}

Return exactly one label: A or B."""


def build_judge_prompt(question: str, response_a: str, response_b: str) -> str:
    return JUDGE_USER_TEMPLATE.format(
        question=question.strip(),
        response_a=response_a.strip(),
        response_b=response_b.strip(),
    )


def build_chat_messages(question: str, response_a: str, response_b: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": build_judge_prompt(question, response_a, response_b),
        },
    ]


def build_sft_example(question: str, response_a: str, response_b: str, label: str) -> dict:
    """Format one preference pair for causal LM SFT (prompt → A|B)."""
    label = label.strip().upper()
    if label not in {"A", "B"}:
        raise ValueError(f"label must be A or B, got {label!r}")
    return {
        "messages": build_chat_messages(question, response_a, response_b)
        + [{"role": "assistant", "content": label}],
        "label": label,
    }
