from minijudge.eval.metrics import majority_vote, position_consistency_metrics


def test_majority_clear():
    assert majority_vote(["A", "A", "B"]) == "A"


def test_majority_all_invalid():
    assert majority_vote([None, None]) is None


def test_position_consistency():
    m = position_consistency_metrics(["A", "B", "A"], ["A", "A", "A"])
    assert m["n_comparable"] == 3
    assert abs(m["position_consistency"] - 2 / 3) < 1e-9
    assert abs(m["conflict_rate"] - 1 / 3) < 1e-9


def test_swap_mapping_logic():
    """After swap, model says A meaning original B; invert restores identity."""
    from minijudge.parser import invert_label

    # Original order: good=A. Swapped order presents original B as A.
    # If judge still prefers the same response, swapped output "A" maps to "B".
    swapped_pred = "A"
    mapped = invert_label(swapped_pred)
    assert mapped == "B"
