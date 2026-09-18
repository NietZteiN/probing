#!/usr/bin/env python
"""E31 (Amendment 4): the value-only chain. Fixed before the runs.

    python scripts/65_simple_chain.py [--models ...] [--level 3] [--seeds 7 11 13]

For the `simple` regime (`cup=5, pen=6`, no equations) and, for comparison, the full chain
(`cot`), reports per (model, target role):

    written-lure excess   the model writes `name=<lure>` at the target's value step, minus the
                          matched neutral twin's rate of writing that digit for the same role
    answer-lure excess    the final answer is the lure, minus the twin's rate of that digit
    accuracy              neutral and incongruent

The written value for a role is the LAST `name=<int>` for that role's name before the first
blank line, exactly the rule the answer parser uses for the queried variable. Three
demonstration seeds; the paper's claim rule (same sign in every seed, pooled cluster bootstrap
over matched sets excluding zero); the pre-registered margin δ = 2 points.

Writes results/summary/simple_chain_L<level>.json.
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

DELTA = 2.0


def written_value(generation: str, name: str) -> int | None:
    block = generation.split("\n\n", 1)[0]
    val = None
    for piece in block.replace("\n", ",").split(","):
        piece = piece.strip()
        if piece.startswith(name + "="):
            rhs = piece[len(name) + 1:].strip()
            if rhs.lstrip("-").isdigit():
                val = int(rhs)
    return val


def load_group(d: Path) -> dict[str, dict]:
    f = d / "behavior.jsonl"
    if not f.exists():
        return {}
    return {r["set_id"]: r for r in map(json.loads, f.open())}


_NAMES: dict[int, dict[str, dict[str, str]]] = {}


def names_of(level: int) -> dict[str, dict[str, str]]:
    """id -> role -> surface name, from the dataset: rows written before 2026-09-14 carry no names."""
    if level not in _NAMES:
        from cueconf.config import DATA_DIR
        _NAMES[level] = {r["id"]: r["names"] for r in map(json.loads, (DATA_DIR / f"L{level}" / "test_sets.jsonl").open())}
    return _NAMES[level]


def cell(model: str, level: int, base: str, role: str, seeds: list[int]) -> dict | None:
    per_seed = {}
    names = names_of(level)
    for sd in seeds:
        d = OUT_DIR / "runs" / model / f"L{level}" / (base if sd == 7 else f"{base}_s{sd}")
        inc, neu = load_group(d / f"incongruent@{role}"), load_group(d / "neutral")
        if not inc or not neu:
            continue
        rows = []
        for sid, r in inc.items():
            t = neu.get(sid)
            if t is None or r["lure"] is None:
                continue
            w_inc = written_value(r["generation"], names[r["id"]][role])
            w_neu = written_value(t["generation"], names[t["id"]][role])
            rows.append({"set": sid, "seed": sd,
                         "wl": float(w_inc == r["lure"]), "wl_twin": float(w_neu == r["lure"]),
                         "al": float(r["pred_is_lure"]), "al_twin": float(t["pred"] == r["lure"]),
                         "acc_inc": float(r["correct"]), "acc_neu": float(t["correct"]),
                         "wrote_any": float(w_inc is not None)})
        if rows:
            per_seed[sd] = rows
    if not per_seed:
        return None
    out = {"seeds": sorted(per_seed), "n_per_seed": {sd: len(v) for sd, v in per_seed.items()}}
    for key, a, b in (("written_lure_excess", "wl", "wl_twin"), ("answer_lure_excess", "al", "al_twin")):
        by_seed = {sd: 100 * float(np.mean([r[a] - r[b] for r in v])) for sd, v in per_seed.items()}
        pooled = np.array([r[a] - r[b] for v in per_seed.values() for r in v])
        sets = np.array([r["set"] for v in per_seed.values() for r in v])
        mean, lo, hi = bootstrap_ci(pooled, sets)
        signs = {np.sign(x) for x in by_seed.values()}
        claim = len(by_seed) >= 3 and len(signs) == 1 and 0 not in signs and (lo > 0 or hi < 0)
        out[key] = {"mean": round(100 * mean, 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2),
                    "by_seed": {str(k): round(v, 2) for k, v in by_seed.items()}, "claimable": bool(claim),
                    "within_delta": bool(-DELTA <= 100 * lo and 100 * hi <= DELTA),
                    "rate_inc": round(100 * float(np.mean([r[a] for v in per_seed.values() for r in v])), 2),
                    "rate_twin": round(100 * float(np.mean([r[b] for v in per_seed.values() for r in v])), 2)}
    out["acc_neutral"] = round(100 * float(np.mean([r["acc_neu"] for v in per_seed.values() for r in v])), 1)
    out["acc_incongruent"] = round(100 * float(np.mean([r["acc_inc"] for v in per_seed.values() for r in v])), 1)
    out["wrote_target_value_step"] = round(100 * float(np.mean([r["wrote_any"] for v in per_seed.values() for r in v])), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    a = ap.parse_args()
    models = a.models or list(load_config("models.yaml")["models"])
    report = {}
    print(f"{'model':16s} {'regime':7s} {'role':4s} {'acc n/i':>11s}  {'written-lure excess [CI]':>28s} {'by seed':>22s}  {'answer-lure excess':>20s}")
    for m in models:
        for base in ("simple", "cot"):
            for role in ("v2", "v1"):
                c = cell(m, a.level, base, role, a.seeds)
                if c is None:
                    continue
                report[f"{m}/L{a.level}/{base}/{role}"] = c
                w, al = c["written_lure_excess"], c["answer_lure_excess"]
                star = "*" if w["claimable"] else (" " if not w["within_delta"] else "=")
                print(f"{m:16s} {base:7s} {role:4s} {c['acc_neutral']:5.1f}/{c['acc_incongruent']:5.1f}  "
                      f"{w['mean']:+6.2f} [{w['lo']:+6.2f},{w['hi']:+6.2f}]{star} "
                      f"{'/'.join(f'{v:+.1f}' for v in w['by_seed'].values()):>22s}  "
                      f"{al['mean']:+6.2f} [{al['lo']:+6.2f},{al['hi']:+6.2f}]{'*' if al['claimable'] else ''}")
    f = RESULTS_DIR / "summary" / f"simple_chain_L{a.level}.json"
    f.write_text(json.dumps(report, indent=1))
    print(f"\n* claimable (same sign in every seed, pooled CI excludes 0); = equivalent to zero (CI inside ±{DELTA:.0f})")
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
