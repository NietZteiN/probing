#!/usr/bin/env python
"""results/summary -> paper/tables/*.tex and paper/numbers.tex. No number in the paper is typed
by hand: main.tex \\input{}s these, and any \\NUM{} key still undefined renders in red.

    python scripts/51_tables.py [--level 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import PROJECT_ROOT, RESULTS_DIR, load_config  # noqa: E402

PAPER = PROJECT_ROOT / "paper"
PRETTY = {"llama32-3b": "Llama-3.2-3B", "llama32-3b-it": "Llama-3.2-3B-Inst.", "llama31-8b": "Llama-3.1-8B",
          "llama31-8b-it": "Llama-3.1-8B-Inst.", "gemma3-4b": "Gemma-3-4B", "olmo2-7b-it": "OLMo-2-7B-Inst.",
          "olmo3-7b-think": "OLMo-3-7B-Think"}


def pct(ci):
    if ci is None:
        return "--"
    p, lo, hi = ci
    return f"{100*p:.1f} {{\\scriptsize[{100*lo:.1f}, {100*hi:.1f}]}}"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--level", type=int, default=3); a = ap.parse_args()
    numbers = {}
    for k in load_config("models.yaml")["models"]:
        for regime in ("cot", "direct"):
            f = RESULTS_DIR / "summary" / k / f"L{a.level}" / regime / "behavior_table.json"
            if not f.exists():
                continue
            t = json.loads(f.read_text())
            for target in ("v2", "v1"):
                inc = t.get(f"incongruent@{target}"); cong = t.get(f"congruent@{target}"); neu = t.get("neutral")
                if not inc:
                    continue
                tag = f"{k.replace('-', '')}{regime}{target}"
                numbers[f"interference{tag}"] = f"{100*t[f'interference@{target}'][0]:.1f}"
                numbers[f"lurerate{tag}"] = f"{100*inc['lure_rate'][0]:.1f}"
    # amendment 2: pooled contrasts across demonstration seeds, when the sweep exists
    sweep = RESULTS_DIR / "summary" / f"seed_sweep_L{a.level}.json"
    if sweep.exists():
        sw = json.loads(sweep.read_text())
        for key, rec in sw.items():
            k, regime = key.split("/")
            tag = f"{k.replace('-', '')}{regime}"
            for name, c in rec["contrasts"].items():
                cname, target = name.split("@")
                numbers[f"{cname}{tag}{target}"] = f"{100*c['pooled_mean']:+.1f}"
                numbers[f"{cname}{tag}{target}lo"] = f"{100*c['ci95'][0]:+.1f}"
                numbers[f"{cname}{tag}{target}hi"] = f"{100*c['ci95'][1]:+.1f}"
            for g, v in rec["groups"].items():
                numbers[f"acc{tag}{g.replace('@', '')}"] = f"{100*v['acc_mean']:.1f}"
                numbers[f"acc{tag}{g.replace('@', '')}lo"] = f"{100*v['acc_range'][0]:.1f}"
                numbers[f"acc{tag}{g.replace('@', '')}hi"] = f"{100*v['acc_range'][1]:.1f}"
            numbers[f"nseeds{tag}"] = str(len(rec["seeds"]))
    (PAPER / "tables").mkdir(exist_ok=True)
    # ---- Table 1 (body): level 3 only, six columns. Other levels go to the appendix table.
    from cueconf.display import MODEL

    def build(levels, path, caption_levels):
        rows = []
        for L in levels:
            sp = RESULTS_DIR / "summary" / f"seed_sweep_L{L}.json"
            if not sp.exists():
                continue
            sw = json.loads(sp.read_text())
            for key, rec in sorted(sw.items()):
                k, regime = key.split("/")
                g, cons = rec["groups"], rec["contrasts"]

                def num(name):
                    c = cons.get(name)
                    if not c:
                        return "--"
                    return f"{100*c['pooled_mean']:+.1f}" + ("$^{*}$" if c["claimable"] else "")
                acc = f"{100*g['neutral']['acc_mean']:.0f}" if "neutral" in g else "--"
                rng = (f"\\,{{\\scriptsize[{100*g['neutral']['acc_range'][0]:.0f},{100*g['neutral']['acc_range'][1]:.0f}]}}"
                       if "neutral" in g else "")
                rows.append([MODEL.get(k, k)] + ([str(L)] if len(levels) > 1 else [])
                            + ["CoT" if regime == "cot" else "direct", acc + rng,
                               num("facilitation@v1"), num("lure_excess@v1"),
                               num("facilitation@v2"), num("lure_excess@v2")])
        if not rows:
            return 0
        ncol = len(rows[0])
        head_lvl = "Level & " if len(levels) > 1 else ""
        with path.open("w") as f:
            f.write("\\begin{tabular}{l" + ("c" if len(levels) > 1 else "") + "lc" + "r" * 4 + "}\n\\toprule\n")
            f.write(f"Model & {head_lvl}Regime & Neutral & \\multicolumn{{2}}{{c}}{{number word on the}} & "
                    f"\\multicolumn{{2}}{{c}}{{number word on the}} \\\\\n")
            f.write(f"& {'& ' if len(levels) > 1 else ''}& accuracy & \\multicolumn{{2}}{{c}}{{\\emph{{queried}} variable}} & "
                    f"\\multicolumn{{2}}{{c}}{{\\emph{{intermediate}} variable}} \\\\\n")
            f.write(f"\\cmidrule(lr){{{ncol-3}-{ncol-2}}}\\cmidrule(lr){{{ncol-1}-{ncol}}}\n")
            f.write(f"& {'& ' if len(levels) > 1 else ''}& (\\%) & helps & lures & helps & lures \\\\\n\\midrule\n")
            prev = None
            for r in rows:
                if prev is not None and r[0] != prev:
                    f.write("\\addlinespace\n")
                prev = r[0]
                f.write(" & ".join(r) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        return len(rows)

    # ---- narrative numbers quoted in the prose, computed here so none is hand-typed
    def narrative(numbers: dict) -> None:
        sweeps = {L: json.loads((RESULTS_DIR / "summary" / f"seed_sweep_L{L}.json").read_text())
                  for L in (1, 2, 3, 4, 5) if (RESULTS_DIR / "summary" / f"seed_sweep_L{L}.json").exists()}
        def scan(kind, regime):
            return [c["pooled_mean"] for sw in sweeps.values() for k, r in sw.items()
                    for n, c in r["contrasts"].items() if n.startswith(kind) and f"/{regime}" in k]
        if sweeps:
            numbers["facilitationmax"] = f"{100*max(scan('facilitation', 'direct')):.0f}"
            numbers["lureexcessmax"] = f"{100*max(scan('lure_excess', 'direct')):.0f}"
            numbers["chain-lure-bound"] = f"{100*max(abs(v) for v in scan('lure_excess', 'cot')):.1f}"
            numbers["n-models"] = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                                   8: "eight"}.get(len({k.split('/')[0] for sw in sweeps.values() for k in sw}), "several")
            numbers["model-list"] = ", ".join(sorted({MODEL.get(k.split("/")[0], k.split("/")[0])
                                                      for sw in sweeps.values() for k in sw}))
        rep = RESULTS_DIR / "summary" / "llama32-3b" / "L3" / "cot" / "replication.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            for v in ("v1", "v2"):
                numbers[f"replication-pre-{v}"] = f"{100*r[v]['acc_pre_cot']:.1f}"
        pg = RESULTS_DIR / "summary" / "llama32-3b" / "L3" / "cot" / "probes_grid.json"
        if pg.exists():
            grid = json.loads(pg.read_text())["train_neutral/v2"]
            def best(pos):
                cells = [v for k, v in grid.items() if k.split("|")[0] == pos and "incongruent@v2" in v]
                return max(cells, key=lambda v: v["neutral"]["accuracy"][0]) if cells else None
            b2, b3, b4 = best("end@v2"), best("query"), best("cotpre@v2")
            if b2:
                numbers["p2-acc"] = f"{b2['neutral']['accuracy'][0]:.2f}"
                numbers["p2-sel"] = f"{b2['neutral']['accuracy'][0] - b2['neutral']['control_acc']:.2f}"
                numbers["p2-lure"] = f"{b2['incongruent@v2']['lure_rate'][0]:.2f}"
            if b3:
                numbers["p3-lure"] = f"{100*b3['incongruent@v2']['lure_rate'][0]:.0f}\\%"
                numbers["p3-ctl"] = f"{b3['neutral']['control_acc']:.2f}"
            if b4:
                numbers["p4-acc"] = f"{b4['neutral']['accuracy'][0]:.2f}"
                numbers["bound-mass"] = f"{b4['incongruent@v2']['lure_mass'][0]:.3f}"
        g4 = RESULTS_DIR / "summary" / "llama32-3b" / "L4" / "cot" / "probes_grid.json"
        if g4.exists():
            gg = json.loads(g4.read_text()).get("train_neutral/v3", {})
            cells = [v for k, v in gg.items() if k.split("|")[0] == "anspre" and "irrelevant@v3" in v]
            if cells:
                numbers["unbound-mass"] = f"{max(cells, key=lambda v: v['neutral']['accuracy'][0])['irrelevant@v3']['lure_mass'][0]:.3f}"
        pt = RESULTS_DIR / "summary" / "llama32-3b" / "L3" / "direct" / "patching.json"
        if pt.exists():
            d = json.loads(pt.read_text())
            m, w, l = (d[f"{c}@v1"]["ALL"]["anspre"] for c in ("main", "ctl_word", "ctl_lure"))
            numbers["lure-removed-3b"] = f"{100*m['lure_removed']:.0f}\\%"
            numbers["nld-3b"] = f"{m['normalized_ld_mean']:.2f}"
            numbers["control-damage"] = f"{100*w['damage']:.0f}\\%"
            numbers["follow-new-lure"] = f"{100*l['follows_src_lure']:.0f}\\%"
        numbers.setdefault("grid-layers", "0--7")

    narrative(numbers)
    n_main = build([3], PAPER / "tables" / "behavior.tex", "3")
    n_app = build([1, 2, 4, 5], PAPER / "tables" / "behavior_levels.tex", "1, 2, 4 and 5")
    print(f"Table 1: {n_main} rows (level 3); appendix table: {n_app} rows")
    with (PAPER / "numbers.tex").open("w") as f:
        f.write("% generated by scripts/51_tables.py -- do not edit\n")
        for key, val in sorted(numbers.items()):
            f.write(f"\\expandafter\\newcommand\\csname NUMval@{key}\\endcsname{{{val}}}\n")
        f.write("\\renewcommand{\\NUM}[1]{\\ifcsname NUMval@#1\\endcsname\\csname NUMval@#1\\endcsname\\else\\textcolor{red}{$\\langle\\langle$\\texttt{#1}$\\rangle\\rangle$}\\fi}\n")
    print(f"wrote {len(numbers)} numbers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
