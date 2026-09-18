#!/usr/bin/env python
"""E21 + E7: two robustness statistics a reviewer is likely to ask for. CPU only.

    python scripts/58_robust_stats.py [--models ...] [--level 3] [--quick]

(1) MIXED-EFFECTS LOGIT beside the cluster bootstrap (E21). Every lure-excess number in the
    paper is a paired per-set difference summarised by a cluster bootstrap. The objection is
    that a bootstrap over matched sets is not a model of the dependence between the two members
    of a twin pair. We refit each cell as

        answered_lure ~ incongruent + C(seed)   with a random intercept per matched set

    on the long form of the same data (one row per set per condition per demonstration seed),
    using a variational Bayes binomial mixed GLM. `incongruent` is 1 on the misleading member of
    the twin and 0 on the neutral member; `answered_lure` is "the model produced the digit that
    the misleading name denotes", which on the neutral member is the pseudo-lure baseline the
    paper already subtracts. The fixed effect on `incongruent` is the lure excess on the
    log-odds scale, so we report it beside the bootstrap's probability-scale estimate and check
    only that the two agree in sign and in what they exclude.

(2) INSTANCE-LEVEL LINK (E7). `50_analysis.py` already fits, per position, a clustered logit of
    a lure error on the probe's margin (log p_true - log p_lure) at the layer with the best
    neutral accuracy, and writes link.json. This script collects those fits across models,
    levels, regimes and roles and asks whether the sign is consistent. A negative coefficient
    means a state that favours the true value produces fewer lure answers, which is what the
    mechanism predicts.

Writes results/summary/robust_stats.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, RESULTS_DIR, load_config  # noqa: E402
from cueconf.stats import bootstrap_ci, mixed_logit  # noqa: E402

SEEDS = (7, 11, 13)


def long_frame(model: str, level: int, base: str, role: str, max_sets: int = 0) -> pd.DataFrame:
    """One row per (matched set, condition, demonstration seed) for the twin pair of `role`.

    `answered_lure` is the same quantity on both members: the misleading member's own lure digit.
    The neutral member therefore carries the pseudo-lure baseline, which is what makes the fixed
    effect on `incongruent` the lure excess rather than a raw lure rate.
    """
    rows = []
    for sd in SEEDS:
        d = OUT_DIR / "runs" / model / f"L{level}" / (base if sd == 7 else f"{base}_s{sd}")
        inc_f, neu_f = d / f"incongruent@{role}" / "behavior.jsonl", d / "neutral" / "behavior.jsonl"
        if not inc_f.exists() or not neu_f.exists():
            continue
        neu = {}
        for line in neu_f.open():
            r = json.loads(line); neu[r["set_id"]] = r
        for line in inc_f.open():
            r = json.loads(line); t = neu.get(r["set_id"])
            if t is None or r.get("lure") is None:
                continue
            rows.append({"set_id": r["set_id"], "seed": sd, "incongruent": 1, "answered_lure": int(r["pred"] == r["lure"])})
            rows.append({"set_id": r["set_id"], "seed": sd, "incongruent": 0, "answered_lure": int(t["pred"] == r["lure"])})
    df = pd.DataFrame(rows)
    if max_sets and len(df):
        sets = np.sort(df["set_id"].unique())
        if len(sets) > max_sets:                      # deterministic subsample: the fit is O(n_sets)
            keep = set(np.random.default_rng(0).choice(sets, max_sets, replace=False))
            df = df[df["set_id"].isin(keep)].reset_index(drop=True)
    return df


def one_cell(df: pd.DataFrame) -> dict:
    """Bootstrap excess and the mixed-effects fixed effect on the same rows."""
    inc = df[df.incongruent == 1].set_index(["set_id", "seed"])["answered_lure"]
    neu = df[df.incongruent == 0].set_index(["set_id", "seed"])["answered_lure"]
    common = inc.index.intersection(neu.index)
    diff = (inc.loc[common] - neu.loc[common]).values.astype(float)
    sets = np.array([s for s, _ in common])
    mean, lo, hi = bootstrap_ci(diff, sets)
    out = {"n_rows": int(len(df)), "n_sets": int(len(set(sets))),
           "boot_excess": round(100 * mean, 2), "boot_lo": round(100 * lo, 2), "boot_hi": round(100 * hi, 2),
           "rate_inc": round(100 * float(inc.loc[common].mean()), 2), "rate_neu": round(100 * float(neu.loc[common].mean()), 2)}
    if df["answered_lure"].nunique() < 2:
        out["mixed"] = {"note": "no variation in the outcome; no model fitted"}
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = mixed_logit(df, "answered_lure ~ incongruent + C(seed)")
            i = list(res.model.exog_names).index("incongruent")
            coef, se = float(res.fe_mean[i]), float(res.fe_sd[i])
            out["mixed"] = {"coef": round(coef, 3), "sd": round(se, 3),
                            "lo": round(coef - 1.96 * se, 3), "hi": round(coef + 1.96 * se, 3),
                            "excludes_zero": bool(abs(coef) > 1.96 * se)}
        except Exception as e:                                   # a cell can be separable
            out["mixed"] = {"note": f"fit failed: {type(e).__name__}: {e}"}
    return out


def agree(cell: dict) -> str | None:
    """Do the bootstrap interval and the mixed-effects interval exclude zero together?"""
    m = cell.get("mixed", {})
    if "coef" not in m:
        return None
    b = cell["boot_lo"] > 0 or cell["boot_hi"] < 0
    return "agree" if b == m["excludes_zero"] else "differ"


def collect_links() -> dict:
    """Every clustered logit of a lure error on the probe margin that 50_analysis.py wrote."""
    fits, per_pos = [], {}
    for f in sorted((RESULTS_DIR / "summary").rglob("link.json")):
        rel = f.relative_to(RESULTS_DIR / "summary")
        model, level, regime = rel.parts[0], rel.parts[1], rel.parts[2]
        for role, positions in json.loads(f.read_text()).items():
            for pos, d in positions.items():
                if "coef_margin" not in d:
                    continue
                kind = pos.split("@")[0]
                rec = {"model": model, "level": level, "regime": regime, "role": role, "position": kind,
                       "layer": d["layer"], "n": d["n"], "coef": round(d["coef_margin"], 3),
                       "se": round(d["se"], 3), "p": d["p"]}
                fits.append(rec)
                per_pos.setdefault(f"{regime.split('_')[0]}/{kind}", []).append(rec)
    summary = {}
    for k, v in sorted(per_pos.items()):
        sig = [r for r in v if r["p"] < 0.05]
        summary[k] = {"n_fits": len(v), "n_sig": len(sig),
                      "n_sig_negative": sum(1 for r in sig if r["coef"] < 0),
                      "n_sig_positive": sum(1 for r in sig if r["coef"] > 0)}
    neg = sum(s["n_sig_negative"] for s in summary.values()); pos = sum(s["n_sig_positive"] for s in summary.values())
    return {"by_regime_position": summary, "fits": fits,
            "total_fits": len(fits), "total_sig": neg + pos, "sig_negative": neg, "sig_positive": pos,
            "sign_consistent": bool(neg == 0 or pos == 0)}


def write(mixed: dict, a, link: dict) -> Path:
    fitted = [c for c in mixed.values() if "coef" in c.get("mixed", {})]
    out = {"level": a.level, "max_sets": a.max_sets, "mixed_effects": mixed, "link": link,
           "mixed_cells": len(mixed), "mixed_fitted": len(fitted),
           "mixed_agree": sum(1 for c in fitted if c["verdict"] == "agree"),
           "mixed_differ": [k for k, c in mixed.items() if c.get("verdict") == "differ"]}
    f = RESULTS_DIR / "summary" / f"robust_stats_L{a.level}.json"
    f.write_text(json.dumps(out, indent=1))
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="one model, for a timing check")
    ap.add_argument("--max-sets", type=int, default=600, help="cap matched sets per cell; the fit cost grows with the number of random effects")
    a = ap.parse_args()
    models = a.models or list(load_config("models.yaml")["models"])
    if a.quick:
        models = models[:1]

    mixed = {}
    for m in models:
        for base in ("cot", "direct"):
            for role in ("v1", "v2"):
                df = long_frame(m, a.level, base, role, a.max_sets)
                if not len(df):
                    continue
                key = f"{m}/L{a.level}/{base}/{role}"
                cell = one_cell(df)
                cell["verdict"] = agree(cell)
                mixed[key] = cell
                mm = cell.get("mixed", {})
                tail = (f"mixed {mm['coef']:+.2f} [{mm['lo']:+.2f}, {mm['hi']:+.2f}] {cell['verdict']}"
                        if "coef" in mm else mm.get("note", ""))
                print(f"{key:44s} boot {cell['boot_excess']:+6.2f} [{cell['boot_lo']:+6.2f}, {cell['boot_hi']:+6.2f}]   {tail}", flush=True)

                write(mixed, a, collect_links())

    fitted = [c for c in mixed.values() if "coef" in c.get("mixed", {})]
    out = {"level": a.level, "max_sets": a.max_sets, "mixed_effects": mixed, "link": collect_links(),
           "mixed_cells": len(mixed), "mixed_fitted": len(fitted),
           "mixed_agree": sum(1 for c in fitted if c["verdict"] == "agree"),
           "mixed_differ": [k for k, c in mixed.items() if c.get("verdict") == "differ"]}
    f = write(mixed, a, out["link"])
    print(f"\nmixed-effects: {out['mixed_agree']}/{out['mixed_fitted']} cells agree with the bootstrap"
          f" ({len(mixed) - len(fitted)} not fitted)")
    if out["mixed_differ"]:
        print("  differ:", ", ".join(out["mixed_differ"]))
    lk = out["link"]
    print(f"link: {lk['total_sig']}/{lk['total_fits']} fits significant at .05; "
          f"{lk['sig_negative']} negative, {lk['sig_positive']} positive; "
          f"sign consistent: {lk['sign_consistent']}")
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
