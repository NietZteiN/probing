#!/usr/bin/env python
"""GPU stage 1 for one model and level: behaviour + hidden-state cache, both regimes.

    python scripts/20_run_model.py --model llama32-3b --level 3 [--regimes cot direct]
        [--groups neutral incongruent@v2 ...] [--no-train] [--layers 0 4 8 ...]

Groups = probe-train splits (train_neutral, train_letter) and test groups keyed
<condition>[@<target>]. Each group writes to $PROBE_OUT/runs/<model>/L<level>/<regime>/<group>/
and is SKIPPED when its summary.json exists, so a killed job resumes by resubmission.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import DATA_DIR, OUT_DIR, model_entry  # noqa: E402
from cueconf.generator import read_jsonl  # noqa: E402
from cueconf.prompts import parse_regime  # noqa: E402
from cueconf.runner import load_model, run_condition  # noqa: E402

MAX_NEW = {"cot": 96, "direct": 12, "free": 320, "simple": 48}


def group_key(x) -> str:
    return f"{x.condition}@{x.target}" if x.target else x.condition


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--regimes", nargs="+", default=["cot", "direct"],
                    help="cot | direct | <regime>_k<N> for N demonstrations instead of 3 (e.g. direct_k16)")
    ap.add_argument("--groups", nargs="*", default=None, help="subset of groups; default all")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap instances per group (smoke tests)")
    ap.add_argument("--all-positions", action="store_true", help="E27: cache every token of the instance region; groups get the suffix __alltok")
    ap.add_argument("--layer-stride", type=int, default=1, help="keep every k-th layer (index 0 = embeddings); 2 halves the cache")
    ap.add_argument("--train-limit", type=int, default=None, help="cap probe-train instances (E27 uses 4000)")
    ap.add_argument("--data-suffix", default="", help="'_ident' to use the identifier-name dataset (E38)")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo groups that already have a summary.json (e.g. a behaviour-only run that now needs caches)")
    ap.add_argument("--no-forced", action="store_true", help="behaviour only: no hidden states, no forced logits (demo-seed sweeps)")
    ap.add_argument("--forced-all", action="store_true",
                    help="also cache hidden states for the *_alt control groups (default: behaviour only for them; "
                         "probes never read them and patching recomputes activations). A labelled cache is 2.3 MB per "
                         "instance for a 3B model (13 positions x 29 layers x 3072 x fp16): 254 GB per (model, level) with them, ~150 GB without")
    a = ap.parse_args()
    m = model_entry(a.model)
    d = DATA_DIR / f"L{a.level}{a.data_suffix}"
    manifest = json.loads((d / "manifest.json").read_text())
    if not manifest["word_pool"]["verified"]:
        print("FATAL: dataset built with UNVERIFIED word lists; run make tokcheck then make data", file=sys.stderr)
        return 2
    groups: dict[str, list] = {}
    if not a.no_train:
        for f in sorted(d.glob("probe_train_*.jsonl")):
            groups["train_" + f.stem.removeprefix("probe_train_")] = read_jsonl(f)
    tests = defaultdict(list)
    for x in read_jsonl(d / "test_sets.jsonl"):
        tests[group_key(x)].append(x)
    groups.update(tests)
    if a.groups:
        groups = {g: groups[g] for g in a.groups}
    tok, model = load_model(m["hf_id"], adapter=m.get("adapter"))
    bs = a.batch_size or m.get("batch_size", 16)
    if a.layer_stride > 1 and a.layers is None:
        from cueconf.runner import text_config
        nl = text_config(model).num_hidden_layers
        a.layers = list(range(0, nl + 1, a.layer_stride))
        if nl not in a.layers:
            a.layers.append(nl)
    suffix = "__alltok" if a.all_positions else ""
    for regime in a.regimes:
        for g, rows in groups.items():
            out = OUT_DIR / "runs" / a.model / f"L{a.level}{a.data_suffix}" / regime / (g + suffix)
            if (out / "summary.json").exists() and not a.overwrite:
                print(f"skip {out} (done)")
                continue
            rows_ = rows[:a.limit] if a.limit else rows
            if g.startswith("train_") and a.train_limit:
                rows_ = rows_[:a.train_limit]
            cond = rows_[0].condition
            s = run_condition(tok, model, a.model, a.level, regime, cond, rows_, out, bs, MAX_NEW[parse_regime(regime)[0]],
                              demo_pool=manifest.get("demo_words") if manifest.get("scheme", "word") != "word" else None,
                              layers=a.layers, do_free=(not g.startswith("train_")) and not a.all_positions,
                              do_forced=(not a.no_forced) and (a.forced_all or "_alt@" not in g), all_positions=a.all_positions)
            print(f"{a.model} L{a.level} {regime} {g}: n={s['n']} acc={s.get('free_accuracy')} lure={s.get('free_lure_rate')} "
                  f"excluded={len(s['excluded'])} {s['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
