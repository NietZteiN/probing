import numpy as np

from cueconf.patch_summary import summarize
from cueconf.patching import layer_sets
from cueconf.stats import bootstrap_ci, crossover_step, fdr, t_star


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    v = rng.binomial(1, 0.7, 1000).astype(float)
    c = np.repeat(np.arange(200), 5)
    p, lo, hi = bootstrap_ci(v, c, n_boot=300)
    assert lo <= p <= hi and abs(p - 0.7) < 0.05 and hi - lo < 0.15


def test_crossover_and_tstar():
    order = ["end@v2", "query", "cotpre@v2", "anspre"]
    assert crossover_step({"end@v2": -1.0, "query": 0.2, "cotpre@v2": -0.1, "anspre": 1.0}, order) == "anspre"
    assert crossover_step({"end@v2": 0.5, "query": 0.2, "cotpre@v2": 0.1, "anspre": 1.0}, order) == "end@v2"
    assert crossover_step({"end@v2": -1.0, "query": -0.2, "cotpre@v2": -0.1, "anspre": -1.0}, order) is None
    assert t_star({-3: 0.2, -1: 0.5, 2: 0.95, 5: 1.0}) == 2
    assert t_star({-3: 0.2}) is None


def test_fdr_rejects_small_pvalues_only():
    rej, adj = fdr(np.array([0.001, 0.01, 0.2, 0.9]))
    assert rej.tolist() == [True, True, False, False]


def test_patching_summary_rates():
    sets = layer_sets(8, window=4)
    names = [s for s, _ in sets]
    assert "ALL" in names and "W0-3" in names and "L7" in names
    rows = [
        {"lure": 2, "src_lure": None, "gold": {"anspre": "6"}, "base": {"anspre": "2"}, "patched": {n: {"anspre": "6"} for n in names}},
        {"lure": 2, "src_lure": None, "gold": {"anspre": "6"}, "base": {"anspre": "2"}, "patched": {n: {"anspre": "2"} for n in names}},
        {"lure": None, "src_lure": None, "gold": {"anspre": "6"}, "base": {"anspre": "6"}, "patched": {n: {"anspre": "6"} for n in names}},
    ]
    s = summarize(rows, sets)["ALL"]["anspre"]
    assert s["n"] == 3 and s["n_lure_err"] == 2 and s["recovery"] == 0.5 and s["damage"] == 0.0
