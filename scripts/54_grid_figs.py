#!/usr/bin/env python
"""Kudo et al. Figs. 5-6 in the cue-conflict setting (E29): heatmap of a patching metric over
equation / chain-step spans (columns, in token order) x layer windows (rows), with the
max-over-windows curve underneath, one panel per source.

    python scripts/54_grid_figs.py --model llama32-3b --level 3 --regime cot --target v2

Sources and the metric drawn:
    other            success_to_source  (Kudo's success rate: the answer becomes the source problem's answer)
    neutral          lure_removed       (share of lure errors no longer answering the lure)
    incongruent_alt  follows_src_lure   (share of answers equal to the patched-in lure)
Read position: the final answer (anspre); use --read cotpre@<target> for the value step.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, PROJECT_ROOT  # noqa: E402
from cueconf.patch_summary import summarize_grid  # noqa: E402

FIG = PROJECT_ROOT / "paper" / "figures"
METRIC = {"other": "success_to_source", "neutral": "lure_removed", "incongruent_alt": "follows_src_lure"}


def seg_order(label: str) -> tuple:
    if label.startswith("in:"):
        return (0, int(label[4:]))
    if label == "query":
        return (1, 0)
    m = re.match(r"cot:(\d+):", label)
    return (2, int(m.group(1)))


def short(label: str) -> str:
    if label.startswith("in:"):
        return f"eq {label[3:]}"
    if label == "query":
        return "query"
    _, k, kind, role = label.split(":")
    return f"{k}:{kind[:4]} {role}"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--regime", default="cot"); ap.add_argument("--target", default="v2"); ap.add_argument("--read", default="anspre")
    a = ap.parse_args()
    base = OUT_DIR / "patching" / a.model / f"L{a.level}" / a.regime
    sources = [s for s in ("other", "neutral", "incongruent_alt") if (base / f"grid_{s}@{a.target}.json").exists()]
    if not sources:
        print("no grid results yet"); return 1
    fig, axes = plt.subplots(2, len(sources), figsize=(4.2 * len(sources), 5.2), gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for j, src in enumerate(sources):
        d = json.loads((base / f"grid_{src}@{a.target}.json").read_text())
        summ = summarize_grid(d["rows"])
        segs = sorted({c.split("|")[0] for c in summ}, key=seg_order)
        wins = sorted({c.split("|")[1] for c in summ}, key=lambda w: int(w[1:].split("-")[0]))
        M = np.full((len(wins), len(segs)), np.nan)
        MIN_N = 20   # a rate over fewer than 20 lure errors is not drawn (1/1 = 1.0 misleads)
        n_den = None
        for c, agg in summ.items():
            s_, w_ = c.split("|")
            v = agg.get(a.read, {}).get(METRIC[src])
            if METRIC[src] == "lure_removed":
                n_den = agg.get(a.read, {}).get("n_lure_err", 0)
                if n_den < MIN_N:
                    v = None
            if v is not None:
                M[wins.index(w_), segs.index(s_)] = v
        ax = axes[0, j]
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(wins))); ax.set_yticklabels(wins, fontsize=7)
        ax.set_xticks(range(len(segs))); ax.set_xticklabels([short(s) for s in segs], rotation=90, fontsize=7)
        n = next(iter(summ.values()))[a.read]["n"] if summ else 0
        extra = f", lure errors={n_den}" if n_den is not None else ""
        ax.set_title(f"source: {src}\n{METRIC[src]} at {a.read} (n={n}{extra})" + (" — too few to draw" if n_den is not None and n_den < MIN_N else ""), fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04)
        ax2 = axes[1, j]
        ax2.plot(range(len(segs)), np.nanmax(M, axis=0), "k.-", lw=1)
        ax2.set_ylim(0, 1.05); ax2.set_xticks(range(len(segs))); ax2.set_xticklabels([short(s) for s in segs], rotation=90, fontsize=7)
        ax2.set_ylabel("max over windows", fontsize=7)
    fig.suptitle(f"{a.model}, level {a.level}, {a.regime}, target {a.target}: span x 4-layer-window patching", fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"kudo_fig5_{a.model}_L{a.level}_{a.regime}_{a.target}"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}.png with sources {sources}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
