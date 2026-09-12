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
from cueconf.patching import run_contrast  # noqa: E402
from cueconf.runner import load_model  # noqa: E402

CONTRASTS = {  # name: (source condition, destination condition)
    "main": ("neutral", "incongruent"),
    "ctl_word": ("neutral_alt", "neutral"),
    "ctl_lure": ("incongruent_alt", "incongruent"),
}


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
    a = ap.parse_args()
    m = model_entry(a.model)
    rows = read_jsonl(DATA_DIR / f"L{a.level}" / "test_sets.jsonl")
    by_set = defaultdict(dict)
    for x in rows:
        by_set[x.set_id][(x.condition, x.target)] = x
    spec = LEVELS[a.level]
    targets = a.targets or [r for r, _ in spec["eqs"] if r != spec["distractor"]]
    tok, model = load_model(m["hf_id"])
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
