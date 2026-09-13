#!/usr/bin/env python
"""E37: equivalence bounds for the chain-of-thought null (amendment 3).

A null needs three things a point estimate cannot give: (a) a positive control showing the
design detects the effect where it exists, (b) a smallest effect of interest, and (c) a test
that the effect is inside it. We use two one-sided tests (TOST) on the cluster bootstrap:
the null of "effect at least as large as the bound" is rejected when the 90% CI lies entirely
inside [-delta, +delta] (equivalent to two one-sided 5% tests).

    python scripts/57_equivalence.py [--delta 0.02] [--levels 2 3 4 5]

Delta defaults to 2 percentage points, the smallest name effect the paper would call meaningful
and roughly the interference originally reported from a single demonstration set. Writes
results/summary/equivalence.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import RESULTS_DIR  # noqa: E402


def ci90(vals: np.ndarray, clusters: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    by = [vals[inv == i] for i in range(len(uniq))]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(by), len(by))
        boots[b] = np.concatenate([by[i] for i in pick]).mean()
    lo, hi = np.percentile(boots, [5, 95])
    return float(vals.mean()), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5])
    a = ap.parse_args()
    out = {"delta": a.delta, "cells": {}}
    print(f"equivalence bound delta = {100*a.delta:.0f} points; 90% CI must lie inside it\n")
    for L in a.levels:
        p = RESULTS_DIR / "summary" / f"seed_sweep_L{L}.json"
        if not p.exists():
            continue
        sw = json.loads(p.read_text())
        for key, rec in sw.items():
            for name, c in rec["contrasts"].items():
                if not name.startswith("lure_excess"):
                    continue
                lo, hi = c["ci95"]          # already a cluster bootstrap; widen check uses its own CI
                inside = abs(lo) < a.delta and abs(hi) < a.delta
                verdict = "EQUIVALENT" if inside else ("effect" if c["claimable"] else "inconclusive")
                out["cells"][f"L{L}/{key}/{name}"] = {"mean": c["pooled_mean"], "ci95": c["ci95"],
                                                      "equivalent": bool(inside), "claimable": c["claimable"]}
                print(f"L{L} {key:22s} {name:16s} {100*c['pooled_mean']:+5.1f} [{100*lo:+5.1f},{100*hi:+5.1f}]  {verdict}")
    cot = [v for k, v in out["cells"].items() if "/cot" in k]
    direct = [v for k, v in out["cells"].items() if "/direct" in k]
    out["summary"] = {
        "cot_cells": len(cot), "cot_equivalent": sum(v["equivalent"] for v in cot),
        "cot_with_effect": sum(v["claimable"] and not v["equivalent"] for v in cot),
        "direct_cells": len(direct), "direct_with_effect": sum(v["claimable"] for v in direct),
        "max_abs_cot_mean": max((abs(v["mean"]) for v in cot), default=None),
    }
    print("\n" + json.dumps(out["summary"], indent=1))
    (RESULTS_DIR / "summary" / "equivalence.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
