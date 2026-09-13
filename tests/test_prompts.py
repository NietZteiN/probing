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


def test_regime_labels_with_demo_counts():
    from cueconf.prompts import parse_regime
    assert parse_regime("cot") == ("cot", 3) and parse_regime("direct") == ("direct", 3)
    assert parse_regime("direct_k16") == ("direct", 16)
    demos = make_demos(3, "word", n=16)
    assert len(demos) == 16 and len({d.input for d in demos}) == 16
    s = next(sample_sets(3, 1, seed=1))
    x = s[1]
    lay = layout(x, demos, "direct_k16")
    assert lay.text.endswith(x.direct) and lay.text[lay.positions["ans"]] == str(x.answer)


def test_segments_tile_the_instance_region():
    demos = make_demos(3, "word")
    for s in sample_sets(3, 5, seed=21):
        for x in s:
            for regime in ("cot", "direct"):
                lay = layout(x, demos, regime)
                segs = lay.segments
                labels = [l for l, _, _ in segs]
                assert labels[:3] == ["in:v1", "in:v2", "query"]
                # every segment reads back as its equation text; segments are ordered and non-overlapping
                for (l, a, b), (l2, a2, b2) in zip(segs, segs[1:]):
                    assert b <= a2
                assert lay.text[segs[0][1]:segs[0][2]] == x.input.split(", ")[0]
                assert lay.text[segs[2][1]:segs[2][2]] == x.input.split("; ")[1]
                if regime == "cot":
                    assert lay.text[segs[3][1]:segs[3][2]] == x.cot.split(", ")[0]
                    assert lay.text[segs[-1][1]:segs[-1][2]] == x.cot.split(", ")[-1]
                    assert segs[-1][0].endswith(f":value:{x.query}")


def test_demo_seed_suffix():
    from cueconf.prompts import demo_seed_of, parse_regime
    assert parse_regime("cot_s11") == ("cot", 3) and demo_seed_of("cot_s11") == 11
    assert parse_regime("direct_k16_s11") == ("direct", 16) and demo_seed_of("direct_k16_s11") == 11
    assert demo_seed_of("cot") == 7
    a = [d.input for d in make_demos(3, "word", seed=7)]; b = [d.input for d in make_demos(3, "word", seed=11)]
    assert a != b and len(b) == 3
