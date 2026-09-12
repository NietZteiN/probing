#!/usr/bin/env python
"""Build every level's probe-train sets and matched test sets (CPU, seconds).

    python scripts/10_build_dataset.py [--levels 3 2 5 4] [--n-test 2000] [--n-train 10000]

Word pool = the INTERSECTION of the rule-4 word lists in data/words/*.json (one per model,
written by 11_tokenizer_check.py inside a job). With no list present the unverified candidate
list is used and the manifest says so; do not run models on such a build.

Outputs data/L<level>/{probe_train_neutral,probe_train_letter,test_sets}.jsonl + manifest.json.
Test sets are equation-level disjoint from BOTH probe-train sets (Kudo et al., footnote 3).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import DATA_DIR, load_config  # noqa: E402
from cueconf.generator import content_hash, expression_split, sample_sets, sample_single, write_jsonl  # noqa: E402
from cueconf.prompts import split_pool  # noqa: E402
from cueconf.words import NEUTRAL_CANDIDATES  # noqa: E402


def word_pool() -> tuple[list[str], dict]:
    lists = sorted((DATA_DIR / "words").glob("*.json"))
    if not lists:
        return list(NEUTRAL_CANDIDATES), {"verified": False, "models": []}
    pool = None
    for f in lists:
        words = set(json.loads(f.read_text())["neutral"])
        pool = words if pool is None else pool & words
    pool = [w for w in NEUTRAL_CANDIDATES if w in pool]      # keep the canonical order
    return pool, {"verified": True, "models": [f.stem for f in lists]}


def main() -> int:
    ap = argparse.ArgumentParser()
    cfg = load_config("dataset.yaml")
    ap.add_argument("--levels", type=int, nargs="+", default=cfg["levels"])
    ap.add_argument("--n-test", type=int, default=cfg["n_test_sets"])
    ap.add_argument("--n-train", type=int, default=cfg["n_probe_train"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    pool, pool_info = word_pool()
    demo_words, inst_words = split_pool(pool)
    if len(inst_words) < 20:
        print(f"FATAL: only {len(inst_words)} instance words after rule 4; widen NEUTRAL_CANDIDATES", file=sys.stderr)
        return 1
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=DATA_DIR.parent).stdout.strip()
    for level in a.levels:
        d = DATA_DIR / f"L{level}"
        if (d / "manifest.json").exists() and not a.force:
            print(f"L{level}: exists, skipping (use --force)")
            continue
        train_exprs, test_exprs = expression_split()
        counts = {}
        for cond in cfg["probe_train_conditions"]:
            rows = list(sample_single(level, a.n_train, cfg["seeds"]["probe_train"], cond, pool=inst_words, allowed_exprs=train_exprs))
            counts[f"probe_train_{cond}"] = write_jsonl(d / f"probe_train_{cond}.jsonl", rows)
        sets = list(sample_sets(level, a.n_test, cfg["seeds"]["test"], pool=inst_words, allowed_exprs=test_exprs))
        flat = [x for s in sets for x in s]
        counts["test_sets"] = write_jsonl(d / "test_sets.jsonl", flat)
        counts["test_matched_sets"] = len(sets)
        groups = {}
        for x in flat:
            g = f"{x.condition}@{x.target}" if x.target else x.condition
            groups[g] = groups.get(g, 0) + 1
        manifest = {"level": level, "built_utc": datetime.now(timezone.utc).isoformat(), "git": sha,
                    "seeds": cfg["seeds"], "counts": counts, "groups": groups, "word_pool": pool_info,
                    "n_instance_words": len(inst_words), "demo_words": demo_words,
                    "expression_holdout": {"n_train": len(train_exprs), "n_test": len(test_exprs), "test": sorted(test_exprs)},
                    "content_hash": content_hash(flat)}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"L{level}: {counts} groups={groups} words={'verified' if pool_info['verified'] else 'UNVERIFIED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
