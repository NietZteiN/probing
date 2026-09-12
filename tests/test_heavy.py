"""Tokenizer- and torch-dependent checks. `pytest --run-heavy` inside a job (make test-full)."""
import json
from pathlib import Path

import numpy as np
import pytest

from cueconf.config import DATA_DIR, load_config, model_entry
from cueconf.generator import sample_sets
from cueconf.prompts import layout, make_demos, scheme_of, tokenize_layout

pytestmark = pytest.mark.heavy


def _tok(key):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_entry(key)["hf_id"])


@pytest.mark.parametrize("key", ["llama32-3b"])
@pytest.mark.parametrize("regime", ["cot", "direct"])
def test_token_layout_constant_across_twins_and_instances(key, regime):
    """Rule 4 at the TOKEN level: every instance in a group has the same length and positions,
    and twins align token for token, using the verified word list for this model."""
    words = DATA_DIR / "words" / f"{key}.json"
    if not words.exists():
        pytest.skip("no word list yet; run make tokcheck")
    pool = json.loads(words.read_text())["neutral"]
    tok = _tok(key)
    demos = make_demos(3, "word")
    sigs = {}
    for s in sample_sets(3, 30, seed=5, pool=pool):
        for x in s:
            if x.condition == "letter":
                continue
            t = tokenize_layout(tok, layout(x, demos, regime))
            assert all(v == [1] for v in t["name_ntok"].values()), (x.id, t["name_ntok"])
            sig = (t["n_tokens"], tuple(sorted(t["positions"].items())))
            sigs.setdefault(x.condition, set()).add(sig)
    for cond, ss in sigs.items():
        assert len(ss) == 1, f"{cond}: {len(ss)} distinct token layouts"
    assert len({next(iter(v)) for v in sigs.values()}) == 1, "conditions differ in layout"


def test_positions_decode_to_expected_characters():
    tok = _tok("llama32-3b")
    demos = make_demos(3, "word")
    s = next(sample_sets(3, 1, seed=2))
    x = next(i for i in s if i.condition == "incongruent" and i.target == "v2")
    lay = layout(x, demos, "cot")
    t = tokenize_layout(tok, lay)
    for label, ti in t["positions"].items():
        if ti is None:
            continue
        piece = tok.decode([t["input_ids"][ti]])
        assert lay.text[lay.positions[label]] in piece, (label, piece)


def test_linear_probe_learns_synthetic_values():
    import torch
    from cueconf.probes import LinearProbe, evaluate
    rng = np.random.default_rng(0)
    d, n = 64, 2000
    centers = rng.normal(size=(10, d))
    y = rng.integers(0, 10, n)
    X = centers[y] + 0.3 * rng.normal(size=(n, d))
    Xt = torch.tensor(X, dtype=torch.float32)
    p = LinearProbe(d, seed=0, device="cpu")
    p.fit_sgd(Xt, torch.tensor(y), lr=1e-2, epochs=500)
    lure = np.where(y == 3, 7, -1)
    ev = evaluate(p, Xt, y, lure)
    assert ev["accuracy"] > 0.95
    assert ev["lure_rate"] < 0.05 and ev["margin_mean"] > 0
