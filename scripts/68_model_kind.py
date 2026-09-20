#!/usr/bin/env python
"""Lure rate by KIND of model, on one spine. CPU only.

    python scripts/68_model_kind.py [--level 3] [--seeds 7 11 13] [--regimes direct cot]

`configs/models.yaml` gives every model a `kind`. On the Llama-3.1-8B spine the ladder is

    base -> instruct -> single-task finetune -> multi-task finetune -> TIES merge -> reasoning

with the three tuned rows being obtune LoRA artefacts folded into the same instruct checkpoint,
so tokenizer, word pool and token positions are identical down the ladder and every row is
comparable instance by instance.

Reports, per (model, regime, target role): neutral accuracy, lure excess over the matched
neutral twin (the paper's contrast), and facilitation. The claim rule is the paper's: same sign
in every demonstration seed and a pooled cluster bootstrap over matched sets excluding zero.

Writes results/summary/model_kind_L<level>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, RESULTS_DIR, load_config  # noqa: E402
from cueconf.stats import bootstrap_ci  # noqa: E402

KIND_ORDER = {"core": 0, "pair": 1, "tuned": 2, "reasoning": 3, "extra": 4}


def load_group(d: Path) -> dict[str, dict]:
    f = d / "behavior.jsonl"
    return {r["set_id"]: r for r in map(json.loads, f.open())} if f.exists() else {}


def cell(model: str, level: int, base: str, role: str, seeds: list[int]) -> dict | None:
    per_seed = {}
    for sd in seeds:
        d = OUT_DIR / "runs" / model / f"L{level}" / (base if sd == 7 else f"{base}_s{sd}")
        inc, neu, con = (load_group(d / g) for g in (f"incongruent@{role}", "neutral", f"congruent@{role}"))
        if not inc or not neu:
            continue
        rows = []
        for sid, r in inc.items():
            t = neu.get(sid)
            if t is None or r["lure"] is None:
                continue
            c = con.get(sid)
            rows.append({"set": sid,
                         "lure": float(r["pred"] == r["lure"]), "pseudo": float(t["pred"] == r["lure"]),
                         "acc_neu": float(t["correct"]), "acc_inc": float(r["correct"]),
                         "acc_con": float(c["correct"]) if c else np.nan})
        if rows:
            per_seed[sd] = rows
    if not per_seed:
        return None
    allr = [r for v in per_seed.values() for r in v]
    sets = np.array([r["set"] for r in allr])
    out = {"seeds": sorted(per_seed), "n": len(allr),
           "acc_neutral": round(100 * float(np.mean([r["acc_neu"] for r in allr])), 1),
           "acc_incongruent": round(100 * float(np.mean([r["acc_inc"] for r in allr])), 1),
           "lure_rate": round(100 * float(np.mean([r["lure"] for r in allr])), 2),
           "pseudo_rate": round(100 * float(np.mean([r["pseudo"] for r in allr])), 2)}
    for key, vals in (("lure_excess", [r["lure"] - r["pseudo"] for r in allr]),
                      ("facilitation", [r["acc_con"] - r["acc_neu"] for r in allr]),
                      ("interference", [r["acc_neu"] - r["acc_inc"] for r in allr])):
        v = np.array(vals, dtype=float)
        ok = ~np.isnan(v)
        if not ok.any():
            continue
        mean, lo, hi = bootstrap_ci(v[ok], sets[ok])
        by_seed = {sd: float(np.mean([(r["lure"] - r["pseudo"]) if key == "lure_excess"
                                      else (r["acc_con"] - r["acc_neu"]) if key == "facilitation"
                                      else (r["acc_neu"] - r["acc_inc"]) for r in rows]))
                   for sd, rows in per_seed.items()}
        by_seed = {k: v_ for k, v_ in by_seed.items() if not np.isnan(v_)}
        signs = {np.sign(x) for x in by_seed.values()}
        out[key] = {"mean": round(100 * mean, 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2),
                    "by_seed": {str(k): round(100 * x, 2) for k, x in by_seed.items()},
                    "claimable": bool(len(by_seed) >= 3 and len(signs) == 1 and 0 not in signs and (lo > 0 or hi < 0))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    ap.add_argument("--regimes", nargs="+", default=["direct", "cot"])
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()
    cfg = load_config("models.yaml")["models"]
    models = a.models or sorted(cfg, key=lambda k: (KIND_ORDER.get(cfg[k].get("kind", "extra"), 9), k))
    report = {}
    for base in a.regimes:
        print(f"\n=== {base}, level {a.level} " + "=" * 52)
        print(f"{'model':22s} {'kind':9s} {'role':4s} {'neu acc':>8s} {'lure':>7s} {'pseudo':>7s} "
              f"{'excess [95% CI]':>24s} {'by seed':>20s}")
        for m in models:
            for role in ("v1", "v2"):
                c = cell(m, a.level, base, role, a.seeds)
                if c is None:
                    continue
                report[f"{m}/L{a.level}/{base}/{role}"] = {"kind": cfg[m].get("kind", "extra"), **c}
                e = c.get("lure_excess", {})
                print(f"{m:22s} {cfg[m].get('kind','?'):9s} {role:4s} {c['acc_neutral']:7.1f}% "
                      f"{c['lure_rate']:6.2f}% {c['pseudo_rate']:6.2f}% "
                      f"{e.get('mean',float('nan')):+7.2f} [{e.get('lo',float('nan')):+6.2f},{e.get('hi',float('nan')):+6.2f}]"
                      f"{'*' if e.get('claimable') else ' '} "
                      f"{'/'.join(f'{v:+.1f}' for v in e.get('by_seed',{}).values()):>20s}")
    f = RESULTS_DIR / "summary" / f"model_kind_L{a.level}.json"
    f.write_text(json.dumps(report, indent=1))
    print(f"\n* same sign in every demonstration seed and pooled CI excluding zero")
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
