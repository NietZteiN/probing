#!/usr/bin/env python
"""Re-parse every stored generation with the current answer parser and rewrite the behaviour
files and summaries in place.

    python scripts/61_reparse.py [--dry-run]

The model outputs are kept verbatim in `behavior.jsonl`, so a parser fix does not need a GPU: it
needs this. Prints every group whose accuracy or lure rate moved, so a silent change is
impossible to miss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import DATA_DIR, OUT_DIR  # noqa: E402
from cueconf.generator import read_jsonl  # noqa: E402
from cueconf.prompts import parse_answer, parse_answer_free, parse_regime  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    # the queried variable's name, per instance id, straight from the dataset: behaviour rows
    # written before 2026-09-14 do not record which role was queried and guessing it is wrong at
    # level 2, where the query is the second variable
    # keyed by DATASET DIRECTORY, not by id alone: data/L3 and data/L3_ident share instance ids,
    # so a global map silently gives identifier names to word-scheme runs (caught 2026-09-14)
    qname: dict[str, dict[str, str]] = {}
    for d in sorted(DATA_DIR.glob("L*")):
        f = d / "test_sets.jsonl"
        if f.exists():
            qname[d.name] = {x.id: x.names[x.query] for x in read_jsonl(f)}
    print("datasets: " + ", ".join(f"{k} ({len(v)})" for k, v in qname.items()))
    changed = n_groups = n_rows = n_flips = 0
    for bf in sorted(OUT_DIR.glob("runs/*/*/*/*/behavior.jsonl")):
        regime, level_dir = bf.parents[1].name, bf.parents[2].name
        parser = parse_answer_free if parse_regime(regime)[0] == "free" else parse_answer
        rows = [json.loads(l) for l in bf.open()]
        n_groups += 1
        out, flips, unresolved = [], 0, []
        for r in rows:
            name = r["names"][r["query"]] if "query" in r else qname.get(level_dir, {}).get(r["id"])
            if name is None:
                unresolved.append(r["id"]); out.append(r); continue
            pred = parser(r["generation"], name)
            correct = pred == r["answer"]
            is_lure = r["lure"] is not None and pred == r["lure"]
            if (pred, correct, is_lure) != (r["pred"], r["correct"], r["pred_is_lure"]):
                flips += 1
            out.append({**r, "pred": pred, "correct": correct, "pred_is_lure": is_lure})
        n_rows += len(rows); n_flips += flips
        if unresolved:
            print(f"  !! {bf.parent.relative_to(OUT_DIR / 'runs')}: {len(unresolved)} rows whose instance is not in any dataset")
        if flips:
            changed += 1
            old_acc = sum(r["correct"] for r in rows) / len(rows)
            new_acc = sum(r["correct"] for r in out) / len(out)
            old_l = sum(r["pred_is_lure"] for r in rows) / len(rows)
            new_l = sum(r["pred_is_lure"] for r in out) / len(out)
            rel = bf.relative_to(OUT_DIR / "runs").parent
            print(f"{str(rel):64s} acc {100*old_acc:5.1f} -> {100*new_acc:5.1f}   lure {100*old_l:4.1f} -> {100*new_l:4.1f}   ({flips} rows)")
            if not a.dry_run:
                with bf.open("w") as f:
                    for r in out:
                        f.write(json.dumps(r) + "\n")
                sf = bf.parent / "summary.json"
                if sf.exists():
                    s = json.loads(sf.read_text())
                    s["free_accuracy"], s["free_lure_rate"] = new_acc, new_l
                    s["reparsed"] = True
                    sf.write_text(json.dumps(s, indent=2))
    print(f"\n{n_groups} groups, {n_rows} rows; {changed} groups changed, {n_flips} rows re-scored"
          + (" (dry run, nothing written)" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
