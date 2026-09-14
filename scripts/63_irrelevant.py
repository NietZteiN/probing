#!/usr/bin/env python
"""Level-4 irrelevant-lure control against its matched neutral baseline, pooled over
demonstration sets: does a number word on the DISTRACTOR produce lure answers above the rate at
which the neutral twin gives that digit? Writes results/summary/irrelevant_L4.json (read by
51_tables.py). No GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import DATA_DIR, OUT_DIR, RESULTS_DIR  # noqa: E402
from cueconf.generator import read_jsonl  # noqa: E402
from cueconf.stats import bootstrap_ci  # noqa: E402


def main() -> int:
    info = {x.id: x for x in read_jsonl(DATA_DIR / "L4" / "test_sets.jsonl")}
    out = {}
    for md in sorted((OUT_DIR / "runs").glob("*/L4")):
        m = md.parent.name
        for base in ("cot", "direct"):
            d_all, sets_all, acc_d, by_seed = [], [], [], {}
            for sd in (7, 11, 13):
                d = md / (base if sd == 7 else f"{base}_s{sd}")
                gi, gn = d / "irrelevant@v3" / "behavior.jsonl", d / "neutral" / "behavior.jsonl"
                if not gi.exists() or not gn.exists():
                    continue
                neu = {}
                for line in gn.open():
                    r = json.loads(line); x = info.get(r["id"])
                    if x:
                        neu[x.set_id] = r
                dd = []
                for line in gi.open():
                    r = json.loads(line); x = info.get(r["id"])
                    t = neu.get(x.set_id) if x else None
                    if x is None or r["lure"] is None or t is None:
                        continue
                    dd.append((1 if r["pred_is_lure"] else 0) - (1 if t["pred"] == r["lure"] else 0))
                    d_all.append(dd[-1]); sets_all.append(x.set_id)
                    acc_d.append((1 if r["correct"] else 0) - (1 if t["correct"] else 0))
                if dd:
                    by_seed[str(sd)] = float(np.mean(dd))
            if not d_all:
                continue
            mean, lo, hi = bootstrap_ci(np.array(d_all, float), np.array(sets_all))
            rec = {"n": len(d_all), "seeds": sorted(by_seed), "lure_excess": float(mean), "ci95": [float(lo), float(hi)],
                   "by_seed": by_seed, "acc_delta": float(np.mean(acc_d)),
                   "claimable": len(by_seed) >= 3 and (all(v > 0 for v in by_seed.values()) or all(v < 0 for v in by_seed.values())) and (lo > 0 or hi < 0)}
            out[f"{m}/{base}"] = rec
            print(f"{m:14s} {base:7s} n={rec['n']} lure excess {100*mean:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] "
                  f"by seed {{{', '.join(f'{k}: {100*v:+.1f}' for k, v in by_seed.items())}}} claimable={rec['claimable']} acc delta {100*rec['acc_delta']:+.1f}")
    (RESULTS_DIR / "summary" / "irrelevant_L4.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
