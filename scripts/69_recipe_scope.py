#!/usr/bin/env python
"""E17 + E18 aggregates, from scratch into results/summary so the paper can cite them. CPU only.

    python scripts/69_recipe_scope.py [--model llama32-3b] [--level 3]

E17 (probe recipe): best-layer neutral accuracy at the positions the paper cites, for the
reported SGD recipe and for the two variants written under `train_neutral__lbfgs` and
`train_neutral__std` by `30_train_probes.py --out-suffix`.

E18 (patching scope): lure removal and damage at the answer position for `--scope all` (the
reported run) and `--scope prompt` (`main@<role>__promptscope.json`), summarised through the
same `patch_summary.summarize` the analysis uses, per layer set.

Writes results/summary/recipe_scope_L<level>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, RESULTS_DIR  # noqa: E402
from cueconf.patch_summary import summarize  # noqa: E402

POSITIONS = ("end@{r}", "query", "cotpre@{r}", "anspre")
RECIPES = {"sgd": "train_neutral", "lbfgs": "train_neutral__lbfgs", "std": "train_neutral__std"}


def best_acc(model: str, level: int, regime: str, sub: str, role: str, pos: str) -> float | None:
    f = OUT_DIR / "probes" / model / f"L{level}" / regime / sub / f"{role}.json"
    if not f.exists():
        return None
    best = 0.0
    for r in json.loads(f.read_text())["results"]:
        if r["position"] != pos:
            continue
        a = r["eval"].get("neutral", {}).get("accuracy")
        if a and a > best:
            best = a
    return best or None


def sets_of(d: dict) -> list[tuple]:
    ls = d["layer_sets"]
    return list(ls.items()) if isinstance(ls, dict) else [(x, None) for x in ls]


def scope_cells(model: str, level: int, role: str, suffix: str) -> dict:
    f = OUT_DIR / "patching" / model / f"L{level}" / "direct" / f"main@{role}{suffix}.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    s = summarize(d["rows"], sets_of(d))
    out = {}
    for k, v in s.items():
        c = (v or {}).get("anspre") or {}
        if c.get("lure_removed") is not None:
            out[k] = {"lure_removed": c["lure_removed"], "damage": c["damage"], "n_lure_err": c.get("n_lure_err")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama32-3b")
    ap.add_argument("--level", type=int, default=3)
    a = ap.parse_args()
    rep: dict = {"model": a.model, "level": a.level, "recipe": {}, "scope": {}}

    for name, sub in RECIPES.items():
        cells = {}
        for role in ("v1", "v2"):
            for tmpl in POSITIONS:
                pos = tmpl.format(r=role)
                acc = best_acc(a.model, a.level, "cot", sub, role, pos)
                if acc is not None:
                    cells[f"{role}/{pos}"] = round(acc, 3)
        if cells:
            rep["recipe"][name] = cells
    for role in ("v1", "v2"):
        for tag, suffix in (("all", ""), ("prompt", "__promptscope")):
            c = scope_cells(a.model, a.level, role, suffix)
            if c:
                rep["scope"][f"{role}/{tag}"] = c

    f = RESULTS_DIR / "summary" / f"recipe_scope_L{a.level}.json"
    f.write_text(json.dumps(rep, indent=1))

    print("E17 probe recipe, best-layer neutral accuracy:")
    poss = sorted({p for c in rep["recipe"].values() for p in c})
    print(f"  {'position':16s} " + " ".join(f"{k:>8s}" for k in RECIPES))
    for p in poss:
        print(f"  {p:16s} " + " ".join(f"{rep['recipe'].get(k,{}).get(p,float('nan')):8.3f}" for k in RECIPES))
    print("\nE18 patching scope, answer position:")
    for k, cells in rep["scope"].items():
        wins = {kk: v for kk, v in cells.items() if kk.startswith("W")}
        best = max(wins.items(), key=lambda kv: kv[1]["lure_removed"]) if wins else (None, {})
        w03 = cells.get("W0-3", {})
        print(f"  {k:12s} best {best[0]} removed {100*best[1].get('lure_removed',0):5.1f}% damage {100*best[1].get('damage',0):5.1f}%"
              f" | W0-3 removed {100*w03.get('lure_removed',0):5.1f}% damage {100*w03.get('damage',0):5.1f}%")
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
