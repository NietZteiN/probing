#!/usr/bin/env python
"""Does the chain protect because it WRITES the value?

For every chain cell we split incongruent instances by whether the model's own chain wrote the
lured variable's true value, and compare the lure rate in the two halves.

THE SPLIT ALONE PROVES NOTHING. A model whose chain breaks down emits some digit, and hits the
lure digit at whatever rate it hits any digit, so the raw lure rate is far higher in the
not-written half for purely arithmetic reasons. Every number below is therefore reported
against the matched neutral twin put through the SAME split: `pseudo` is how often the twin
(whose name carries no value) answers that same digit. The claim is the EXCESS, lure - pseudo.

No GPU: the generations are stored verbatim in behavior.jsonl.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import DATA_DIR, OUT_DIR, RESULTS_DIR  # noqa: E402
from cueconf.generator import read_jsonl  # noqa: E402
from cueconf.prompts import parse_regime  # noqa: E402
from cueconf.stats import bootstrap_ci  # noqa: E402


def wrote_value(text: str, name: str, value: int) -> bool:
    """True if the chain states `name=<value>` anywhere before the blank line: the model wrote
    the variable's own true value, which is the event the mechanism says does the work."""
    block = text.split("\n\n", 1)[0]
    return re.search(rf"\b{re.escape(name)}\s*=\s*{value}\b", block) is not None


def main() -> int:
    rows = []
    for L in (1, 2, 3, 4, 5):
        f = DATA_DIR / f"L{L}" / "test_sets.jsonl"
        if not f.exists():
            continue
        info = {x.id: x for x in read_jsonl(f)}
        for bf in sorted((OUT_DIR / "runs").glob(f"*/L{L}/*/incongruent@*/behavior.jsonl")):
            model, regime, group = bf.parents[3].name, bf.parents[1].name, bf.parent.name
            if parse_regime(regime)[0] != "cot":
                continue
            target = group.split("@")[1]
            nf = bf.parent.parent / "neutral" / "behavior.jsonl"
            if not nf.exists():
                continue
            neu = {}
            for line in nf.open():
                r = json.loads(line); x = info.get(r["id"])
                if x is not None:
                    neu[x.set_id] = (r, x)
            acc = {True: {"n": 0, "lure": 0, "nc": 0, "pseudo": 0, "sets": [], "d": []},
                   False: {"n": 0, "lure": 0, "nc": 0, "pseudo": 0, "sets": [], "d": []}}
            for line in bf.open():
                r = json.loads(line); x = info.get(r["id"])
                if x is None or r["lure"] is None:
                    continue
                w = wrote_value(r["generation"], x.names[target], x.values[target])
                a = acc[w]; a["n"] += 1; a["lure"] += 1 if r["pred_is_lure"] else 0
                tw = neu.get(x.set_id)
                if tw is not None:
                    nr, nx = tw
                    # the twin must fall in the SAME half, or the baseline is not comparable
                    if wrote_value(nr["generation"], nx.names[target], nx.values[target]) == w:
                        a["nc"] += 1; a["pseudo"] += 1 if nr["pred"] == r["lure"] else 0
                        a["sets"].append(x.set_id)
                        a["d"].append((1 if r["pred_is_lure"] else 0) - (1 if nr["pred"] == r["lure"] else 0))
            if acc[True]["n"] + acc[False]["n"] == 0:
                continue
            import numpy as np
            rec = {"model": model, "level": L, "regime": regime, "target": target,
                   "frac_wrote": acc[True]["n"] / (acc[True]["n"] + acc[False]["n"])}
            for w, tag in ((True, "wrote"), (False, "not")):
                a = acc[w]
                rec[f"n_{tag}"] = a["n"]
                rec[f"lure_{tag}"] = a["lure"] / a["n"] if a["n"] else None
                rec[f"pseudo_{tag}"] = a["pseudo"] / a["nc"] if a["nc"] else None
                rec[f"excess_{tag}"] = (rec[f"lure_{tag}"] - rec[f"pseudo_{tag}"]) if a["nc"] and a["n"] else None
                rec[f"excess_{tag}_ci"] = (bootstrap_ci(np.array(a["d"]), np.array(a["sets"]))[1:]
                                           if len(a["d"]) >= 50 else None)
            rows.append(rec)
    out = RESULTS_DIR / "summary" / "value_written.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    import numpy as np
    hdr = (f"{'model':14s} {'L':2s} {'regime':8s} {'tgt':3s} {'wrote%':>7s} | "
           f"{'lure':>6s} {'pseudo':>7s} {'EXCESS':>7s} (written) | {'lure':>6s} {'pseudo':>7s} {'EXCESS':>7s} (not)")
    print(hdr); print("-" * len(hdr))
    def f(v, w=6):
        return f"{100*v:{w}.2f}" if v is not None else " " * (w - 2) + "--"
    for r in sorted(rows, key=lambda r: r["frac_wrote"]):
        print(f"{r['model']:14s} {r['level']:<2d} {r['regime']:8s} {r['target']:3s} "
              f"{100*r['frac_wrote']:6.1f}% | {f(r['lure_wrote'])} {f(r['pseudo_wrote'],7)} "
              f"{f(r['excess_wrote'],7)}            | {f(r['lure_not'])} {f(r['pseudo_not'],7)} {f(r['excess_not'],7)}")
    print()
    for tag, label in (("wrote", "the chain WROTE the value"), ("not", "it did NOT")):
        sel = [r for r in rows if r[f"n_{tag}"] >= 50 and r[f"excess_{tag}"] is not None]
        if not sel:
            continue
        wts = [r[f"n_{tag}"] for r in sel]
        print(f"pooled over {len(sel):3d} cells where {label:27s}: "
              f"raw lure {100*np.average([r[f'lure_{tag}'] for r in sel], weights=wts):5.2f}%  "
              f"pseudo {100*np.average([r[f'pseudo_{tag}'] for r in sel], weights=wts):5.2f}%  "
              f"EXCESS {100*np.average([r[f'excess_{tag}'] for r in sel], weights=wts):+5.2f} pts")
    big = [r for r in rows if r.get("excess_wrote") and r["n_wrote"] >= 200
           and r["excess_wrote_ci"] and r["excess_wrote_ci"][0] > 0.01]
    print(f"\ncells with a real lure excess DESPITE the value being written: {len(big)}")
    for r in sorted(big, key=lambda r: -r["excess_wrote"]):
        ci = r["excess_wrote_ci"]
        print(f"   {r['model']:14s} L{r['level']} {r['regime']:8s} {r['target']} "
              f"excess {100*r['excess_wrote']:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] (n={r['n_wrote']})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
