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
from cueconf.display import MODEL, REGIME, position, rcparams  # noqa: E402

FIG = PROJECT_ROOT / "paper" / "figures"


def heatmap(ax, grid: dict, cond: str, metric: str, positions: list[str], title: str):
    layers = sorted({int(k.split("|L")[1]) for k in grid})
    M = np.full((len(layers), len(positions)), np.nan)
    for key, conds in grid.items():
        pos, l = key.split("|L")
        if pos in positions and cond in conds and conds[cond].get(metric):
            M[layers.index(int(l)), positions.index(pos)] = conds[cond][metric][0]
    if M.size == 0 or np.all(np.isnan(M)):
        ax.set_title(title + " (no data)", fontsize=8); return None
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu" if metric.startswith("margin") else "viridis",
                   vmin=(-np.nanmax(np.abs(M)) if metric.startswith("margin") else 0), vmax=(np.nanmax(np.abs(M)) if metric.startswith("margin") else 1))
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([position(p_) for p_ in positions], rotation=30, ha="right")
    ax.set_ylabel("layer"); ax.set_title(title)
    return im


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(rcparams())
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--level", type=int, default=3); ap.add_argument("--role", default="v2")
    ap.add_argument("--patch-regime", default="direct", help="regime for Figure 3 (lure errors exist mainly without a chain)")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    r = a.role
    positions = [f"def@{r}", f"end@{r}", "query", f"cotpre@{r}", "anspre"]
    # ---- Fig 2
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, regime in zip(axes, ("cot", "direct")):
        f = RESULTS_DIR / "summary" / a.model / f"L{a.level}" / regime / "probes_grid.json"
        if not f.exists():
            ax.set_title(f"{regime}: no probes yet"); continue
        grid = json.loads(f.read_text()).get(f"train_neutral/{r}", {})
        pos = positions if regime == "cot" else [p for p in positions if not p.startswith("cotpre")]
        im = heatmap(ax, grid, f"incongruent@{r}", "margin_mean", pos, REGIME.get(regime, regime))
        if im is not None:
            cb = fig.colorbar(im, ax=ax, fraction=0.045)
        cb.set_label("log p(true value) - log p(lure value)")
    fig.suptitle(f"{MODEL.get(a.model, a.model)}: does the state hold the true value or the lure?", y=1.02)
    for ax in axes:
        ax.set_xlabel("position in the problem and its solution")
    fig.tight_layout(); fig.savefig(FIG / f"fig2_margin_{a.model}_L{a.level}_{r}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"fig2_margin_{a.model}_L{a.level}_{r}.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    # ---- Fig 3
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    f = RESULTS_DIR / "summary" / a.model / f"L{a.level}" / a.patch_regime / "patching.json"
    if f.exists():
        pt = json.loads(f.read_text())
        # main: lure removed among lure errors (and normalised LD); ctl_word: damage; ctl_lure: follows the new lure
        for name, key, style, lab in ((f"main@{r}", "lure_removed", "-", "lure removed (neutral patch)"),
                                      (f"main@{r}", "normalized_ld_mean", "-.", "normalised logit diff."),
                                      (f"ctl_word@{r}", "damage", "--", "damage (word control)"),
                                      (f"ctl_lure@{r}", "follows_src_lure", ":", "follows new lure (lure control)")):
            if name not in pt:
                continue
            xs, ys = [], []
            for lset, agg in pt[name].items():
                if lset.startswith("L") and "anspre" in agg and agg["anspre"].get(key) is not None:
                    xs.append(int(lset[1:])); ys.append(agg["anspre"][key])
            if xs:
                ax.plot(xs, ys, style, label=lab)
        ax.set_xlabel("layer whose activations were replaced")
        ax.set_ylabel("share of cases"); ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{MODEL.get(a.model, a.model)}, {REGIME.get(a.patch_regime, a.patch_regime)}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
    else:
        ax.set_title("no patching yet")
    fig.tight_layout(); fig.savefig(FIG / f"fig3_patching_{a.model}_L{a.level}_{r}.pdf"); fig.savefig(FIG / f"fig3_patching_{a.model}_L{a.level}_{r}.png", dpi=160); plt.close(fig)
    print(f"figures -> {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
