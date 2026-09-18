#!/usr/bin/env python
"""E33 / amendment 2: behavioural quantities across demonstration seeds.

    python scripts/56_seed_sweep.py [--models ...] [--levels 2 3 4 5] [--seeds 7 11 13]

Regime directories: `cot` / `direct` (seed 7) and `cot_s11`, `direct_s13`, ... For every
(model, level, regime, group) reports accuracy and lure rate per seed, the mean and range across
seeds, and for the contrasts (interference, facilitation, lure minus pseudo-lure) whether the
sign holds in every seed plus a pooled cluster bootstrap over matched sets with seeds as
replicates. Writes results/summary/seed_sweep_L<level>.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, RESULTS_DIR, load_config  # noqa: E402
from cueconf.stats import bootstrap_ci  # noqa: E402


def load(run_dir: Path) -> pd.DataFrame:
    rows = []
    for g in sorted(run_dir.iterdir()):
        f = g / "behavior.jsonl"
        if f.exists() and "__" not in g.name:
            for line in f.open():
                d = json.loads(line)
                rows.append({"group": g.name, "set_id": d["set_id"], "correct": int(d["correct"]), "lure": d["lure"],
                             "pred": d["pred"], "pred_is_lure": int(d["pred_is_lure"]), "target": d["target"]})
    return pd.DataFrame(rows)


def contrasts(df: pd.DataFrame) -> dict:
    """Per-set differences for the pre-registered contrasts, as arrays keyed by contrast."""
    out = {}
    neu = df[df.group == "neutral"].set_index("set_id")
    for t in sorted({g.split("@")[1] for g in df.group.unique() if "@" in g and g.startswith("incongruent@")}):
        inc = df[df.group == f"incongruent@{t}"].set_index("set_id"); cong = df[df.group == f"congruent@{t}"].set_index("set_id")
        common = neu.index.intersection(inc.index).intersection(cong.index)
        out[f"interference@{t}"] = (neu.loc[common, "correct"] - inc.loc[common, "correct"]).values.astype(float)
        out[f"facilitation@{t}"] = (cong.loc[common, "correct"] - neu.loc[common, "correct"]).values.astype(float)
        pseudo = (neu.loc[common, "pred"].values == inc.loc[common, "lure"].values).astype(float)
        out[f"lure_excess@{t}"] = inc.loc[common, "pred_is_lure"].values.astype(float) - pseudo
        out["_sets"] = np.asarray(common)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    # default to every model in the panel: a short default silently shrank the sweep, and with it
    # Table 1 and Figure 1, from eight models to two on 2026-09-14
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    ap.add_argument("--data-suffix", default="", help="'_ident' for the identifier-name runs (E38)")
    a = ap.parse_args()
    models = a.models or list(load_config("models.yaml")["models"])
    for L in a.levels:
        report = {}
        for m in models:
            for base in ("cot", "direct", "simple"):
                per_seed = {}
                for sd in a.seeds:
                    d = OUT_DIR / "runs" / m / f"L{L}{a.data_suffix}" / (base if sd == 7 else f"{base}_s{sd}")
                    if d.exists():
                        df = load(d)
                        if len(df):
                            per_seed[sd] = df
                if not per_seed:
                    continue
                rec = {"seeds": sorted(per_seed), "groups": {}, "contrasts": {}}
                groups = sorted(set.intersection(*[set(df.group.unique()) for df in per_seed.values()]))
                for g in groups:
                    accs = {sd: float(df[df.group == g].correct.mean()) for sd, df in per_seed.items()}
                    lures = {sd: float(df[df.group == g].pred_is_lure.mean()) for sd, df in per_seed.items()}
                    rec["groups"][g] = {"acc_by_seed": accs, "acc_mean": float(np.mean(list(accs.values()))),
                                        "acc_range": [min(accs.values()), max(accs.values())],
                                        "lure_by_seed": lures, "lure_mean": float(np.mean(list(lures.values())))}
                cons = {sd: contrasts(df) for sd, df in per_seed.items()}
                for name in sorted({k for c in cons.values() for k in c if not k.startswith("_")}):
                    vals = {sd: float(c[name].mean()) for sd, c in cons.items() if name in c}
                    pooled = np.concatenate([c[name] for c in cons.values() if name in c])
                    clusters = np.concatenate([np.array([f"{sd}:{s}" for s in c["_sets"]]) for sd, c in cons.items() if name in c])
                    # cluster on matched SET across seeds so a set's three replicates move together
                    set_clusters = np.concatenate([c["_sets"] for c in cons.values() if name in c])
                    ci = bootstrap_ci(pooled, set_clusters, n_boot=1000)
                    # amendment 2(b): the sign must hold in EVERY seed; a seed at exactly zero (ceiling) fails it
                    signs = {np.sign(v) for v in vals.values()}
                    same = len(signs) == 1 and 0 not in signs
                    rec["contrasts"][name] = {"by_seed": vals, "pooled_mean": ci[0], "ci95": ci[1:], "same_sign_all_seeds": same,
                                              "claimable": same and (ci[1] > 0 or ci[2] < 0)}
                report[f"{m}/{base}"] = rec
                print(f"=== L{L} {m} {base}: seeds {rec['seeds']} ===")
                for g in ("neutral", "letter"):
                    if g in rec["groups"]:
                        r = rec["groups"][g]; print(f"  {g:16s} acc {100*r['acc_mean']:5.1f} range [{100*r['acc_range'][0]:.1f}, {100*r['acc_range'][1]:.1f}]")
                for name, c in rec["contrasts"].items():
                    print(f"  {name:18s} pooled {100*c['pooled_mean']:+5.1f} [{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}] by seed " +
                          " ".join(f"{sd}:{100*v:+.1f}" for sd, v in c["by_seed"].items()) + ("  CLAIMABLE" if c["claimable"] else ""))
        (RESULTS_DIR / "summary" / f"seed_sweep_L{L}{a.data_suffix}.json").write_text(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
