#!/usr/bin/env python
"""GPU stage 3: activation patching for one (model, level, regime).

    python scripts/40_patch.py --model llama32-3b --level 3 --regime cot
        [--contrasts main ctl_word ctl_lure] [--targets v1 v2] [--scope all|prompt] [--limit N]
        [--only-lure-errors]   # restrict main to instances the model got wrong with the lure (from behavior.jsonl)

Output: $PROBE_OUT/patching/<model>/L<level>/<regime>/<contrast>@<target>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import DATA_DIR, OUT_DIR, model_entry  # noqa: E402
from cueconf.generator import LEVELS, read_jsonl  # noqa: E402
from cueconf.patching import run_contrast, run_grid_contrast  # noqa: E402
from cueconf.runner import load_model  # noqa: E402

CONTRASTS = {  # name: (source condition, destination condition)
    "main": ("neutral", "incongruent"),
    "ctl_word": ("neutral_alt", "neutral"),
    "ctl_lure": ("incongruent_alt", "incongruent"),
}


def _with_target(x, t):
    import dataclasses
    return dataclasses.replace(x, target=t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--regime", default="cot")
    ap.add_argument("--contrasts", nargs="+", default=list(CONTRASTS))
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--scope", default="all", choices=["all", "prompt"])
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-lure-errors", action="store_true")
    ap.add_argument("--grid", action="store_true", help="E29: Kudo-style equation-span x layer-window grid instead of name-position sweeps")
    ap.add_argument("--sources", nargs="+", default=["other", "neutral", "incongruent_alt"],
                    help="grid sources: other = a different neutral problem into neutral (Kudo's design); neutral / incongruent_alt = twins into incongruent")
    a = ap.parse_args()
    m = model_entry(a.model)
    rows = read_jsonl(DATA_DIR / f"L{a.level}" / "test_sets.jsonl")
    by_set = defaultdict(dict)
    for x in rows:
        by_set[x.set_id][(x.condition, x.target)] = x
    spec = LEVELS[a.level]
    targets = a.targets or [r for r, _ in spec["eqs"] if r != spec["distractor"]]
    tok, model = load_model(m["hf_id"])
    if a.grid:
        sets = list(by_set.values())
        for srcname in a.sources:
            for t in targets:
                pairs = []
                if srcname == "other":
                    # a different neutral problem with a different answer, cyclically paired
                    neutrals = [d[("neutral", None)] for d in sets if ("neutral", None) in d]
                    for i, x in enumerate(neutrals):
                        y = neutrals[(i + 1) % len(neutrals)]
                        if y.answer != x.answer:
                            pairs.append((y, x))
                    # the "target" for reads is t (which value step to read)
                    pairs = [(s_, d_) for s_, d_ in pairs]
                else:
                    # the neutral twin carries no target; every other twin is keyed by the renamed slot
                    for d in sets:
                        s_ = d.get(("neutral", None)) if srcname == "neutral" else d.get((srcname, t))
                        d_ = d.get(("incongruent", t))
                        if s_ and d_:
                            pairs.append((s_, d_))
                if not pairs:
                    print(f"grid_{srcname}@{t}: no pairs, skipping"); continue
                if a.limit:
                    pairs = pairs[:a.limit]
                out = OUT_DIR / "patching" / a.model / f"L{a.level}" / a.regime / f"grid_{srcname}@{t}.json"
                if out.exists():
                    print(f"skip {out}"); continue
                if srcname == "other":
                    # read positions need a target: set it on copies of the neutral instances
                    import copy
                    pairs = [(copy.replace(s_, target=t) if hasattr(copy, "replace") else _with_target(s_, t), _with_target(d_, t)) for s_, d_ in pairs]
                s = run_grid_contrast(tok, model, a.level, a.regime, pairs, f"grid_{srcname}@{t}", out, window=a.window)
                print(f"grid_{srcname}@{t}: {len(s)} cells", flush=True)
        return 0
    for cname in a.contrasts:
        src_c, dst_c = CONTRASTS[cname]
        for t in targets:
            pairs = []
            for sid, d in by_set.items():
                src = d.get((src_c, t)) or d.get((src_c, None))
                dst = d.get((dst_c, t)) or d.get((dst_c, None))
                if src and dst:
                    pairs.append((src, dst))
            if a.limit:
                pairs = pairs[:a.limit]
            only = None
            if a.only_lure_errors and cname == "main":
                beh = OUT_DIR / "runs" / a.model / f"L{a.level}" / a.regime / f"{dst_c}@{t}" / "behavior.jsonl"
                only = {json.loads(l)["id"] for l in beh.open() if json.loads(l)["pred_is_lure"]}
            out = OUT_DIR / "patching" / a.model / f"L{a.level}" / a.regime / f"{cname}@{t}.json"
            if out.exists():
                print(f"skip {out}"); continue
            s = run_contrast(tok, model, a.level, a.regime, pairs, f"{cname}@{t}", out, scope=a.scope, window=a.window, only_ids=only)
            print(f"{cname}@{t}: ALL-layers summary {json.dumps(s.get('ALL'), default=float)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
