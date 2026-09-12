#!/usr/bin/env python
"""What exists: datasets, word lists, cached runs, probes, patching. Safe on the login node."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import DATA_DIR, OUT_DIR, load_config  # noqa: E402


def main() -> int:
    print(f"data:    {DATA_DIR}")
    for m in sorted(DATA_DIR.glob("L*/manifest.json")):
        d = json.loads(m.read_text())
        print(f"  {m.parent.name}: {d['counts']} words={'verified' if d['word_pool']['verified'] else 'UNVERIFIED'} hash={d['content_hash']}")
    words = sorted((DATA_DIR / "words").glob("*.json"))
    print(f"words:   {[w.stem for w in words] or 'none (run make tokcheck)'}")
    print(f"out:     {OUT_DIR}")
    models = load_config("models.yaml")["models"]
    for key in models:
        runs = sorted((OUT_DIR / "runs" / key).glob("L*/*/*/summary.json"))
        probes = sorted((OUT_DIR / "probes" / key).glob("L*/*/*/*.json"))
        patches = sorted((OUT_DIR / "patching" / key).glob("L*/*/*.json"))
        if runs or probes or patches:
            print(f"  {key}: runs={len(runs)} probes={len(probes)} patch={len(patches)}")
            for r in runs:
                s = json.loads(r.read_text())
                acc = s.get("free_accuracy"); lr = s.get("free_lure_rate")
                print(f"    {r.parent.relative_to(OUT_DIR / 'runs' / key)}: n={s['n']} acc={acc} lure={lr} excl={len(s['excluded'])}")
        else:
            print(f"  {key}: nothing yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
