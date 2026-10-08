from cueconf.followups import fixed_sample, value_boundary


def test_boundary_does_not_mistake_operands_for_written_value():
    text = "cup=2 + 3, cup=5, pen=1 + cup, pen=1 + 5, pen=6"
    boundary = value_boundary(text, "cup")
    assert boundary["value"] == 5
    assert text[boundary["marker"]:boundary["value_start"]] == "="
    boundary = value_boundary("cup=2 + 3 = 5, pen=6", "cup")
    assert boundary["value"] == 5
    assert value_boundary("cup=2 + 3, pen=6", "cup") is None
    assert value_boundary("cup=5, cup=7", "cup")["value"] == 5


def test_fixed_sample_is_order_independent_and_nested():
    ids = [f"set-{i}" for i in range(20)]
    assert fixed_sample(ids, 5) == fixed_sample(reversed(ids), 5)
    assert fixed_sample(ids, 5) <= fixed_sample(ids, 10)
