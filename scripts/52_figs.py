#!/usr/bin/env python
"""Figures from results/summary: Fig 2 margin heatmap (layers x positions, CoT vs direct),
Fig 3 patching recovery by layer with the control patch, and the appendix accuracy heatmaps.

    python scripts/52_figs.py --model llama32-3b [--level 3] [--role v2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

FIG = PROJECT_ROOT / "paper" / "figures"


def heatmap(ax, grid: dict, cond: str, metric: str, positions: list[str], title: str):
    layers = sorted({int(k.split("|L")[1]) for k in grid})
    M = np.full((len(layers), len(positions)), np.nan)
    for key, conds in grid.items():
        pos, l = key.split("|L")
        if pos in positions and cond in conds and conds[cond].get(metric):
            M[layers.index(int(l)), positions.index(pos)] = conds[cond][metric][0]
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu" if metric.startswith("margin") else "viridis",
                   vmin=(-np.nanmax(np.abs(M)) if metric.startswith("margin") else 0), vmax=(np.nanmax(np.abs(M)) if metric.startswith("margin") else 1))
    ax.set_xticks(range(len(positions))); ax.set_xticklabels(positions, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("layer"); ax.set_title(title, fontsize=9)
    return im


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--level", type=int, default=3); ap.add_argument("--role", default="v2")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    r = a.role
    positions = [f"def@{r}", f"end@{r}", "query", f"cotpre@{r}", "anspre"]
    # ---- Fig 2
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    for ax, regime in zip(axes, ("cot", "direct")):
        f = RESULTS_DIR / "summary" / a.model / f"L{a.level}" / regime / "probes_grid.json"
        if not f.exists():
            ax.set_title(f"{regime}: no probes yet"); continue
        grid = json.loads(f.read_text()).get(f"train_neutral/{r}", {})
        pos = positions if regime == "cot" else [p for p in positions if not p.startswith("cotpre")]
        im = heatmap(ax, grid, f"incongruent@{r}", "margin_mean", pos, f"{a.model} {regime}: log p(true) - log p(lure)")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(FIG / f"fig2_margin_{a.model}_L{a.level}_{r}.pdf"); plt.close(fig)
    # ---- Fig 3
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    f = RESULTS_DIR / "summary" / a.model / f"L{a.level}" / "cot" / "patching.json"
    if f.exists():
        pt = json.loads(f.read_text())
        for name, style in ((f"main@{r}", "-"), (f"ctl_word@{r}", "--"), (f"ctl_lure@{r}", ":")):
            if name not in pt:
                continue
            xs, ys = [], []
            for lset, agg in pt[name].items():
                if lset.startswith("L") and "anspre" in agg:
                    v = agg["anspre"]["recovery"] if name.startswith("main") else (agg["anspre"]["damage"] if "word" in name else agg["anspre"]["follows_src_lure"])
                    if v is not None:
                        xs.append(int(lset[1:])); ys.append(v)
            if xs:
                ax.plot(xs, ys, style, label=name)
        ax.set_xlabel("patched layer"); ax.set_ylabel("rate"); ax.legend(fontsize=6); ax.set_ylim(0, 1)
    else:
        ax.set_title("no patching yet")
    fig.tight_layout(); fig.savefig(FIG / f"fig3_patching_{a.model}_L{a.level}_{r}.pdf"); plt.close(fig)
    print(f"figures -> {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
