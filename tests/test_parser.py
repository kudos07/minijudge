from minijudge.parser import invert_label, parse_ab_label


def test_exact_a():
    r = parse_ab_label("A")
    assert r.valid and r.label == "A"


def test_exact_b_with_noise():
    r = parse_ab_label("  B\n")
    assert r.valid and r.label == "B"


def test_response_a_phrase():
    r = parse_ab_label("Response A is better.")
    assert r.valid and r.label == "A"


def test_invalid():
    r = parse_ab_label("I cannot decide between them.")
    assert not r.valid and r.label is None


def test_invert():
    assert invert_label("A") == "B"
    assert invert_label("B") == "A"
    assert invert_label(None) is None
