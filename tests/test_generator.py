"""Rules 1-3 and 6 from PLAN.md §4.3, plus the Kudo chain format, on every level."""
import itertools
import random

import pytest

from cueconf.generator import (LEVELS, NUMBER_WORD_CONDITIONS, Arithmetic, Equation, cot_steps,
                               legal_lures, render_cot, render_input, sample_arithmetic,
                               sample_sets, sample_single)
from cueconf.words import NUMBER_WORDS, WORD_TO_DIGIT

LEVELS_ALL = sorted(LEVELS)


def _sets(level, n=40, seed=1):
    return list(sample_sets(level, n, seed))


def test_kudo_table1_chains():
    # Level 3 running example from the paper: A=1+B, B=2+3; A=?
    ar = Arithmetic(3, [Equation("v1", "+", ("1", "v2")), Equation("v2", "+", ("2", "3"))], "v1", {"v1": 6, "v2": 5})
    names = {"v1": "A", "v2": "B"}
    assert render_input(ar, names) == "A=1 + B, B=2 + 3; A=?"
    assert render_cot(ar, names) == "A=1 + B, B=2 + 3, B=5, A=1 + B, A=1 + 5, A=6"
    # Level 1: A=1+B, B=2
    ar1 = Arithmetic(1, [Equation("v1", "+", ("1", "v2")), Equation("v2", None, ("2",))], "v1", {"v1": 3, "v2": 2})
    assert render_cot(ar1, names) == "A=1 + B, B=2, A=1 + B, A=1 + 2, A=3"
    # Level 2: A=2+3, B=1+A; B=?
    ar2 = Arithmetic(2, [Equation("v1", "+", ("2", "3")), Equation("v2", "+", ("1", "v1"))], "v2", {"v1": 5, "v2": 6})
    assert render_cot(ar2, names) == "B=1 + A, A=2 + 3, A=5, B=1 + A, B=1 + 5, B=6"
    # Level 5: A=1+B, B=2+C, C=1+2
    n3 = {"v1": "A", "v2": "B", "v3": "C"}
    ar5 = Arithmetic(5, [Equation("v1", "+", ("1", "v2")), Equation("v2", "+", ("2", "v3")), Equation("v3", "+", ("1", "2"))], "v1", {"v1": 6, "v2": 5, "v3": 3})
    assert render_cot(ar5, n3) == "A=1 + B, B=2 + C, C=1 + 2, C=3, B=2 + C, B=2 + 3, B=5, A=1 + B, A=1 + 5, A=6"
    # Level 4: distractor C is omitted from the chain
    ar4 = Arithmetic(4, [Equation("v1", "+", ("1", "v2")), Equation("v2", "+", ("2", "3")), Equation("v3", "+", ("4", "5"))], "v1", {"v1": 6, "v2": 5, "v3": 9})
    assert render_input(ar4, n3) == "A=1 + B, B=2 + 3, C=4 + 5; A=?"
    assert render_cot(ar4, n3) == "A=1 + B, B=2 + 3, B=5, A=1 + B, A=1 + 5, A=6"


@pytest.mark.parametrize("level", LEVELS_ALL)
def test_values_single_digit(level):
    rng = random.Random(0)
    for _ in range(200):
        ar = sample_arithmetic(level, rng)
        assert all(0 <= v <= 9 for v in ar.values.values())
        assert len(ar.eqs) == len(LEVELS[level]["eqs"])


@pytest.mark.parametrize("level", LEVELS_ALL)
def test_rule1_exactly_one_number_word(level):
    for s in _sets(level):
        for x in s:
            n_num = sum(1 for nm in x.names.values() if nm in WORD_TO_DIGIT)
            if x.condition in NUMBER_WORD_CONDITIONS:
                assert n_num == 1, x
                assert x.target is not None and x.names[x.target] in WORD_TO_DIGIT
            else:
                assert n_num == 0, x
                assert x.target is None or x.condition == "neutral_alt"


@pytest.mark.parametrize("level", LEVELS_ALL)
def test_rule2_lure_never_correct_for_another_reason(level):
    for s in _sets(level):
        for x in s:
            if x.condition in ("incongruent", "incongruent_alt", "irrelevant"):
                lure = WORD_TO_DIGIT[x.names[x.target]]
                assert x.lure == lure
                assert lure not in x.values.values(), x               # not any variable's value
                digits = {int(a) for e in x.eqs for a in e["args"] if a.isdigit()}
                assert lure not in digits, x                          # not any operand digit
                assert 0 <= lure <= 9                                 # rule 3
                assert x.lure_distance == abs(lure - x.values[x.target])
            if x.condition == "congruent":
                assert WORD_TO_DIGIT[x.names[x.target]] == x.values[x.target]
                assert x.lure is None


@pytest.mark.parametrize("level", LEVELS_ALL)
def test_rule6_twins_share_arithmetic(level):
    for s in _sets(level):
        base = s[0]
        for x in s[1:]:
            assert x.set_id == base.set_id
            assert x.eqs == base.eqs and x.values == base.values and x.answer == base.answer
        # every non-letter twin differs from the neutral twin in at most one name slot
        neutral = next(x for x in s if x.condition == "neutral")
        for x in s:
            if x.condition in ("letter", "neutral"):
                continue
            diff = [r for r in neutral.names if neutral.names[r] != x.names[r]]
            assert diff == [x.target], (x.condition, diff)
        # and the incongruent / alt lures differ
        for t in {x.target for x in s if x.target}:
            inc = [x for x in s if x.target == t and x.condition == "incongruent"]
            alt = [x for x in s if x.target == t and x.condition == "incongruent_alt"]
            if inc and alt:
                assert inc[0].lure != alt[0].lure


def test_targets_cover_query_and_intermediate_at_level3():
    s = _sets(3, n=1)[0]
    conds = sorted((x.condition, x.target) for x in s)
    assert ("incongruent", "v1") in conds and ("incongruent", "v2") in conds
    assert ("congruent", "v1") in conds and ("congruent", "v2") in conds
    assert not any(x.condition == "irrelevant" for x in s)


def test_irrelevant_only_with_distractor():
    s = _sets(4, n=1)[0]
    irr = [x for x in s if x.condition == "irrelevant"]
    assert len(irr) == 1 and irr[0].target == "v3"
    # the queried chain keeps neutral names
    assert all(irr[0].names[r] not in WORD_TO_DIGIT for r in ("v1", "v2"))


def test_names_unique_within_instance():
    for level in LEVELS_ALL:
        for s in _sets(level, n=20):
            for x in s:
                assert len(set(x.names.values())) == len(x.names)


def test_train_test_disjoint_at_equation_level():
    """Kudo et al. footnote 3: an expression seen in probe training never appears in a test instance."""
    from cueconf.generator import ALL_EXPRESSIONS, expression_split, instance_arithmetic
    tr, te = expression_split()
    assert tr.isdisjoint(te) and tr | te == set(ALL_EXPRESSIONS)
    assert len(ALL_EXPRESSIONS) > 100
    train = list(sample_single(3, 500, seed=45, condition="neutral"))
    test = list(sample_sets(3, 100, seed=44))
    train_exprs = set().union(*(instance_arithmetic(x).expressions() for x in train))
    test_exprs = set().union(*(instance_arithmetic(s[0]).expressions() for s in test))
    assert train_exprs <= tr and test_exprs <= te
    assert train_exprs.isdisjoint(test_exprs)
    # uniqueness at the rendered-text level
    assert len({x.input for x in train}) == len(train)
    assert len({s[1].input for s in test}) == len(test)


def test_determinism():
    a = [x.to_json() for s in sample_sets(3, 5, seed=3) for x in s]
    b = [x.to_json() for s in sample_sets(3, 5, seed=3) for x in s]
    assert a == b


def test_legal_lures_excludes_values_and_digits():
    ar = Arithmetic(3, [Equation("v1", "+", ("1", "v2")), Equation("v2", "+", ("2", "3"))], "v1", {"v1": 6, "v2": 5})
    assert legal_lures(ar, "v2") == [0, 4, 7, 8, 9]
