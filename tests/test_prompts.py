"""Positions P1-P5 on characters (no tokenizer needed), for both regimes and every level."""
import pytest

from cueconf.generator import LEVELS, sample_sets
from cueconf.prompts import (N_DEMOS, layout, make_demos, parse_answer, scheme_of, split_pool)
from cueconf.words import NEUTRAL_CANDIDATES


@pytest.mark.parametrize("level", sorted(LEVELS))
@pytest.mark.parametrize("regime", ["cot", "direct"])
def test_positions_point_at_the_right_characters(level, regime):
    demos = {s: make_demos(level, s) for s in ("letter", "word")}
    for s in sample_sets(level, 15, seed=9):
        for x in s:
            lay = layout(x, demos[scheme_of(x.condition)], regime)
            t = lay.text
            p = lay.positions
            assert t.startswith(lay.prompt) and t.endswith(lay.output)
            assert t[p["query"]] == "?"
            assert t[p["anspre"]] == "=" and t[p["ans"]] == str(x.answer)
            for r, nm in x.names.items():
                assert t[p[f"def@{r}"]:p[f"def@{r}"] + len(nm)] == nm
                e = p[f"end@{r}"]
                assert t[e].isalnum() and t[e + 1] in ",;"          # last operand char
                if p[f"use@{r}"] is not None:
                    assert t[p[f"use@{r}"]:p[f"use@{r}"] + len(nm)] == nm
                if regime == "cot" and p[f"cotpre@{r}"] is not None:
                    assert t[p[f"cotpre@{r}"]] == "="
                    assert t[p[f"cot@{r}"]] == str(x.values[r])
                if regime == "direct":
                    assert p[f"cotpre@{r}"] is None
            if regime == "cot":
                assert p[f"cotpre@{x.query}"] == p["anspre"]
                d = LEVELS[level]["distractor"]
                if d:
                    assert p[f"cotpre@{d}"] is None                 # distractor not in the chain
            # every recorded name span really is that name, and none lies in the demos
            for r, spans in lay.name_spans.items():
                assert spans, r
                for a, b in spans:
                    assert t[a:b] == x.names[r]
                    assert a >= lay.instance_start


def test_demos_fixed_and_disjoint_from_instance_pool():
    d1 = make_demos(3, "word"); d2 = make_demos(3, "word")
    assert [x.input for x in d1] == [x.input for x in d2] and len(d1) == N_DEMOS
    demo_words, inst_words = split_pool()
    assert not set(demo_words) & set(inst_words)
    assert set(demo_words) | set(inst_words) == set(NEUTRAL_CANDIDATES)
    for x in d1:
        assert set(x.names.values()) <= set(demo_words)
    letters = make_demos(3, "letter")
    assert all(all(len(n) == 1 and n.isupper() for n in x.names.values()) for x in letters)


def test_layout_is_identical_across_twins_up_to_names():
    """Rule 4 at the character level: twins differ only inside name spans."""
    demos = make_demos(3, "word")
    for s in sample_sets(3, 10, seed=11):
        neutral = next(x for x in s if x.condition == "neutral")
        ln = layout(neutral, demos, "cot")
        for x in s:
            if x.condition == "letter":
                continue
            lx = layout(x, demos, "cot")
            # same number of positions, same order of labels
            assert list(lx.positions) == list(ln.positions)


def test_parse_answer():
    assert parse_answer("pen=1 + two, two=2 + 3, two=5, pen=1 + two, pen=1 + 5, pen=6\n\nfoo", "pen") == 6
    assert parse_answer("pen=6", "pen") == 6
    assert parse_answer("pen=1 + two, two=2", "pen") is None
    assert parse_answer("pen=-1", "pen") == -1
    assert parse_answer("cup=5, pen=7, pen=8", "pen") == 8
