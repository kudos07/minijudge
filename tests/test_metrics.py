from minijudge.data.bias_suite import BIAS_TYPES, build_bias_examples
from minijudge.eval.metrics import compute_classification_metrics
from minijudge.prompts import build_judge_prompt, build_sft_example


def test_metrics_perfect():
    m = compute_classification_metrics(["A", "B", "A"], ["A", "B", "A"])
    assert m["accuracy"] == 1.0
    assert m["invalid_output_rate"] == 0.0


def test_metrics_invalid():
    m = compute_classification_metrics(["A", "B"], ["A", None])
    assert m["n_valid"] == 1
    assert m["invalid_output_rate"] == 0.5


def test_bias_suite_size():
    rows = build_bias_examples()
    assert len(rows) == 25 * len(BIAS_TYPES)
    assert all(r["label"] in {"A", "B"} for r in rows)
    types = {r["bias_type"] for r in rows}
    assert types == set(BIAS_TYPES)


def test_prompt_contains_parts():
    p = build_judge_prompt("Q?", "ans A", "ans B")
    assert "Response A:" in p and "Response B:" in p
    assert "Q?" in p


def test_sft_example():
    ex = build_sft_example("Q", "A1", "B1", "B")
    assert ex["messages"][-1]["content"] == "B"
