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
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, RESULTS_DIR  # noqa: E402
from cueconf.stats import bootstrap_ci  # noqa: E402

sweep = importlib.import_module("56_seed_sweep")


def ci90(vals: np.ndarray, clusters: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    return bootstrap_ci(vals, clusters, n_boot=n_boot, seed=seed, confidence=0.90)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    out = {"delta": a.delta, "confidence": 0.90, "n_boot": 4000,
           "cluster": "matched set across demonstration seeds", "cells": {}}
    print(f"equivalence bound delta = {100*a.delta:.0f} points; 90% CI must lie inside it\n")
    for L in a.levels:
        p = RESULTS_DIR / "summary" / f"seed_sweep_L{L}.json"
        if not p.exists():
            continue
        sw = json.loads(p.read_text())
        for key, rec in sw.items():
            model, regime = key.split("/")
            if regime not in ("cot", "direct"):
                continue
            per_seed = {}
            for sd in rec["seeds"]:
                run = OUT_DIR / "runs" / model / f"L{L}" / (regime if sd == 7 else f"{regime}_s{sd}")
                per_seed[sd] = sweep.contrasts(sweep.load(run))
            for name, c in rec["contrasts"].items():
                if not name.startswith("lure_excess") and regime != "cot":
                    continue
                vals = np.concatenate([s[name] for s in per_seed.values() if name in s])
                clusters = np.concatenate([s["_sets"] for s in per_seed.values() if name in s])
                mean, lo, hi = ci90(vals, clusters)
                if not np.isclose(mean, c["pooled_mean"]):
                    raise ValueError(f"{key}/{name}: raw mean disagrees with seed sweep")
                inside = lo > -a.delta and hi < a.delta
                verdict = "EQUIVALENT" if inside else ("effect" if c["claimable"] else "inconclusive")
                out["cells"][f"L{L}/{key}/{name}"] = {"mean": mean, "ci90": [lo, hi], "ci95": c["ci95"],
                                                      "n": len(vals), "n_sets": len(np.unique(clusters)),
                                                      "seeds": rec["seeds"], "contrast": name.split("@")[0],
                                                      "equivalent": bool(inside), "claimable": c["claimable"]}
                print(f"L{L} {key:22s} {name:16s} {100*c['pooled_mean']:+5.1f} [{100*lo:+5.1f},{100*hi:+5.1f}]  {verdict}")
    cot = [v for k, v in out["cells"].items() if "/cot/" in k and v["contrast"] == "lure_excess"]
    direct = [v for k, v in out["cells"].items() if "/direct" in k]
    accuracy = [v for k, v in out["cells"].items() if "/cot/" in k and v["contrast"] != "lure_excess"]
    out["summary"] = {
        "cot_cells": len(cot), "cot_equivalent": sum(v["equivalent"] for v in cot),
        "cot_with_effect": sum(v["claimable"] and not v["equivalent"] for v in cot),
        "direct_cells": len(direct), "direct_with_effect": sum(v["claimable"] for v in direct),
        "max_abs_cot_mean": max((abs(v["mean"]) for v in cot), default=None),
        "cot_accuracy_cells": len(accuracy),
        "cot_accuracy_equivalent": sum(v["equivalent"] for v in accuracy),
        "cot_accuracy_claimable": sum(v["claimable"] for v in accuracy),
        "cot_accuracy_claimable_not_equivalent": sum(v["claimable"] and not v["equivalent"] for v in accuracy),
    }
    print("\n" + json.dumps(out["summary"], indent=1))
    (RESULTS_DIR / "summary" / "equivalence.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
