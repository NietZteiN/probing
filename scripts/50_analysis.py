#!/usr/bin/env python
"""Aggregate everything into results/summary/ (small JSON/CSV; safe on the login node).

    python scripts/50_analysis.py [--models ...] [--level 3]

Produces, per model/level/regime:
  behavior.csv        one row per instance (condition, target, correct, pred_is_lure, ...)
  behavior_table.json accuracy, lure rate, interference, facilitation with cluster-bootstrap CIs
  probes_grid.json    per (role, position, layer): accuracy / lure_rate / lure_mass / margin
                      per condition, mean and spread over seeds; control accuracy
  crossover.json      first chain position where margin > 0 and stays > 0 (per layer-max)
  link.json           logistic regression: lure error ~ margin at P2/P3/P4 (+ model, level)
  patching.json       recovery / damage / follows-new-lure per layer set
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, RESULTS_DIR, load_config  # noqa: E402
from cueconf.stats import bootstrap_ci, clustered_logit, crossover_step, paired_difference_ci  # noqa: E402

CHAIN_ORDER = ["def@{r}", "end@{r}", "query", "cotpre@{r}", "anspre"]


def behavior_frame(run_dir: Path) -> pd.DataFrame:
    rows = []
    for g in sorted(run_dir.iterdir()):
        f = g / "behavior.jsonl"
        if f.exists():
            for line in f.open():
                d = json.loads(line); d["group"] = g.name; rows.append(d)
    return pd.DataFrame(rows)


def behavior_table(df: pd.DataFrame) -> dict:
    out = {}
    for grp, sub in df.groupby("group"):
        acc = bootstrap_ci(sub["correct"].values.astype(float), sub["set_id"].values)
        lure = bootstrap_ci(sub["pred_is_lure"].values.astype(float), sub["set_id"].values) if sub["lure"].notna().any() else None
        out[grp] = {"n": int(len(sub)), "accuracy": acc, "lure_rate": lure}
    for t in sorted({g.split("@")[1] for g in df["group"].unique() if "@" in g}):
        for name, a, b in (("interference", "neutral", f"incongruent@{t}"), ("facilitation", f"congruent@{t}", "neutral")):
            if a in df["group"].values and b in df["group"].values:
                d2 = df.rename(columns={"group": "condition"})
                out[f"{name}@{t}"] = paired_difference_ci(d2, a, b)
    return out


def probe_grid(probe_json: Path) -> dict:
    d = json.loads(probe_json.read_text())
    grid = defaultdict(lambda: defaultdict(list))
    ctl = defaultdict(lambda: defaultdict(list))
    for r in d["results"]:
        key = f"{r['position']}|L{r['layer']}"
        for cond, ev in r["eval"].items():
            grid[key][cond].append(ev)
        for cond, v in r.get("control_acc", {}).items():
            ctl[key][cond].append(v)
    out = {}
    for key, conds in grid.items():
        out[key] = {}
        for cond, evs in conds.items():
            agg = {}
            for m in ("accuracy", "lure_rate", "lure_mass", "margin_mean", "margin_pos_frac"):
                vals = [e[m] for e in evs if e.get(m) is not None]
                agg[m] = (float(np.mean(vals)), float(np.std(vals))) if vals else None
            if ctl[key].get(cond):
                agg["control_acc"] = float(np.mean(ctl[key][cond]))
                agg["selectivity"] = agg["accuracy"][0] - agg["control_acc"]
            out[key][cond] = agg
    return {"role": d["role"], "grid": out}


def crossover(grid: dict, role: str, cond: str) -> dict:
    """Max-over-layers margin per chain position, then the crossover step."""
    order = [p.format(r=role) for p in CHAIN_ORDER]
    best = {}
    for key, conds in grid.items():
        pos, layer = key.split("|")
        if cond in conds and conds[cond].get("margin_mean"):
            m = conds[cond]["margin_mean"][0]
            best[pos] = max(best.get(pos, -1e9), m)
    return {"order": order, "max_margin": best, "crossover": crossover_step(best, order)}


def link_regression(model: str, level: int, regime: str, role: str, beh: pd.DataFrame, probe_json: Path, layer_pick: str = "best") -> dict:
    """Does the margin at end@r / query / cotpre@r (best layer, seed-mean) predict a lure error?"""
    d = json.loads(probe_json.read_text())
    npz = np.load(probe_json.with_suffix(".npz"))
    cond = f"incongruent@{role}"
    ids = d["test_ids"].get(cond)
    if ids is None:
        return {}
    sub = beh[beh["group"] == cond].set_index("id").loc[ids]
    out = {}
    for pos in (f"end@{role}", "query", f"cotpre@{role}"):
        # pick the layer with the highest neutral accuracy at this position
        cands = [(r["layer"], r["eval"].get("neutral", {}).get("accuracy", 0)) for r in d["results"] if r["position"] == pos]
        if not cands:
            continue
        layer = max(set(l for l, _ in cands), key=lambda l: np.mean([a for ll, a in cands if ll == l]))
        keys = [k for k in npz.files if k.startswith(f"{pos}/L{layer}/") and k.endswith(f"/{cond}/margin")]
        if not keys:
            continue
        margin = np.nanmean(np.stack([npz[k] for k in keys]), axis=0)
        df = pd.DataFrame({"lure_err": sub["pred_is_lure"].values.astype(int), "margin": margin, "set_id": sub["set_id"].values})
        df = df.dropna()
        if df["lure_err"].nunique() < 2:
            out[pos] = {"layer": int(layer), "note": "no variation in lure errors", "n": int(len(df))}
            continue
        res = clustered_logit(df, "lure_err ~ margin")
        out[pos] = {"layer": int(layer), "n": int(len(df)), "coef_margin": float(res.params["margin"]),
                    "se": float(res.bse["margin"]), "p": float(res.pvalues["margin"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--level", type=int, default=3)
    a = ap.parse_args()
    models = a.models or list(load_config("models.yaml")["models"])
    for k in models:
        for regime in ("cot", "direct"):
            run_dir = OUT_DIR / "runs" / k / f"L{a.level}" / regime
            if not run_dir.exists():
                continue
            out = RESULTS_DIR / "summary" / k / f"L{a.level}" / regime
            out.mkdir(parents=True, exist_ok=True)
            beh = behavior_frame(run_dir)
            if len(beh):
                beh.drop(columns=["generation"]).to_csv(out / "behavior.csv", index=False)
                (out / "behavior_table.json").write_text(json.dumps(behavior_table(beh), indent=1))
            grids, xover, links = {}, {}, {}
            for pj in sorted((OUT_DIR / "probes" / k / f"L{a.level}" / regime).glob("*/*.json")):
                g = probe_grid(pj); role = g["role"]; tr = pj.parent.name
                grids[f"{tr}/{role}"] = g["grid"]
                for cond in (f"incongruent@{role}", f"irrelevant@{role}"):
                    if any(cond in c for c in g["grid"].values()):
                        xover[f"{tr}/{role}/{cond}"] = crossover(g["grid"], role, cond)
                if len(beh) and tr == "train_neutral":
                    links[role] = link_regression(k, a.level, regime, role, beh, pj)
            if grids:
                (out / "probes_grid.json").write_text(json.dumps(grids))
                (out / "crossover.json").write_text(json.dumps(xover, indent=1))
                (out / "link.json").write_text(json.dumps(links, indent=1))
            pt = {}
            for pj in sorted((OUT_DIR / "patching" / k / f"L{a.level}" / regime).glob("*.json")):
                pt[pj.stem] = json.loads(pj.read_text())["summary"]
            if pt:
                (out / "patching.json").write_text(json.dumps(pt, indent=1, default=float))
            print(f"{k} L{a.level} {regime}: behavior={len(beh)} probes={len(grids)} patching={len(pt)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
