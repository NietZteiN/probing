#!/usr/bin/env python
"""Figure 1: the whole paper in one picture.

    python scripts/60_master_figure.py [--level 3]

Three panels, left to right:
  (a) the manipulation, on one problem: the same arithmetic with a neutral name, a name that
      agrees with the variable's value, and a name that contradicts it;
  (b) the headline, per model: how much a name changes the answer, with and without a chain of
      thought. Top strip: accuracy gained when the name agrees. Bottom strip: how much more
      often the model answers the name's own value when it lies. The grey band is the
      equivalence bound of two points that the chain-of-thought results sit inside;
  (c) inside the model, at every token of one problem: how often a probe trained on neutral
      problems reads the variable's true value, and how often it reads the value its name
      denotes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from cueconf.display import MODEL, rcparams  # noqa: E402

FIG = PROJECT_ROOT / "paper" / "figures"
GREEN, RED, GREY, BLUE = "#157A55", "#B5321F", "#5B6470", "#2B4A9E"


# models grouped by family, small to large, so a reader can scan families and sizes
FAMILIES = [("Llama", ["llama32-3b", "llama32-3b-it", "llama31-8b", "llama31-8b-it"]),
            ("Gemma", ["gemma3-4b-it", "gemma3-12b-it"]),
            ("OLMo", ["olmo2-1b-it", "olmo2-7b-it"])]
SHORT = {"llama32-3b": "Llama-3.2-3B", "llama32-3b-it": "Llama-3.2-3B Inst.",
         "llama31-8b": "Llama-3.1-8B", "llama31-8b-it": "Llama-3.1-8B Inst.",
         "gemma3-4b-it": "Gemma-3-4B", "gemma3-12b-it": "Gemma-3-12B",
         "olmo2-1b-it": "OLMo-2-1B", "olmo2-7b-it": "OLMo-2-7B"}


def layout_models(sw: dict):
    """(row y, model key, label, family, is_family_start) for models present, grouped by family."""
    have = {k.split("/")[0] for k in sw}
    rows, y = [], 0.0
    for fam, keys in FAMILIES:
        present = [k for k in keys if k in have]
        if not present:
            continue
        for i, k in enumerate(present):
            rows.append((y, k, SHORT.get(k, k), fam, i == 0))
            y += 1
        y += 0.6                      # gap between families
    return rows


def paired(ax, sw: dict, rows, contrast: str, title: str, xlabel: str, delta: float,
           band: bool, ylabels: bool):
    """One row per model; the two regimes on the same row, joined so the change is one gesture."""
    if band:
        ax.axvspan(-100 * delta, 100 * delta, color="0.90", zorder=0, lw=0)
    ax.axvline(0, color="0.45", lw=0.9, zorder=1)
    for y, key, lab, fam, first in rows:
        pts = {}
        for regime in ("cot", "direct"):
            c = sw.get(f"{key}/{regime}", {}).get("contrasts", {}).get(contrast)
            if c:
                pts[regime] = (100 * c["pooled_mean"], 100 * c["ci95"][0], 100 * c["ci95"][1])
        if len(pts) == 2:
            ax.plot([pts["cot"][0], pts["direct"][0]], [y, y], color="0.72", lw=1.4, zorder=2)
        for regime, colour, marker in (("cot", BLUE, "s"), ("direct", RED, "o")):
            if regime not in pts:
                continue
            m, lo, hi = pts[regime]
            ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt=marker, ms=5.5, color=colour,
                        ecolor=colour, elinewidth=1.1, capsize=2, zorder=3)
    ys = [r[0] for r in rows]
    ax.set_yticks(ys)
    ax.set_yticklabels([r[2] for r in rows], fontsize=9)
    ax.tick_params(axis="y", pad=2, length=0, labelleft=ylabels)
    ax.set_ylim(max(ys) + 0.9, min(ys) - 0.9)          # first family at the top
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.grid(axis="x", color="0.93", lw=0.7, zorder=0)
    for fam, keys in FAMILIES:
        fam_rows = [r for r in rows if r[3] == fam]
        if fam_rows and fam_rows[0][0] > 0:
            ax.axhline(fam_rows[0][0] - 0.8, color="0.85", lw=0.8, zorder=0)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3); ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--probe-model", default="llama32-3b"); ap.add_argument("--role", default="v1")
    a = ap.parse_args()
    plt.rcParams.update(rcparams())
    sw = json.loads((RESULTS_DIR / "summary" / f"seed_sweep_L{a.level}.json").read_text())

    fig = plt.figure(figsize=(16.4, 4.7))
    gs = GridSpec(1, 4, width_ratios=[0.98, 0.98, 0.86, 1.70], wspace=0.20, figure=fig)
    gs.update(left=0.005, right=0.995)
    axa = fig.add_subplot(gs[:, 0]); axa.axis("off")
    axb1 = fig.add_subplot(gs[:, 1])
    axb2 = fig.add_subplot(gs[:, 2])          # not sharey: a shared formatter would blank both
    axc = fig.add_subplot(gs[:, 3])

    # ---- (a) the manipulation
    axa.set_title("(a)  the manipulation", loc="left", fontsize=11)
    rows_a = [("neutral", "truck=2 + ant, ant=7 - 6; truck=?", GREY,
               "answer 3"),
              ("the name agrees", "three=2 + ant, ant=7 - 6; three=?", GREEN,
               "answer 3; the name says 3"),
              ("the name lies", "four=2 + ant, ant=7 - 6; four=?", RED,
               "answer 3; the name says 4")]
    y = 0.88
    for lab, prob, col, note in rows_a:
        axa.text(0, y, lab, fontsize=10.5, color=col, weight="bold", transform=axa.transAxes)
        axa.text(0, y - 0.09, prob, fontsize=8.6, family="monospace", color=col, transform=axa.transAxes)
        axa.text(0, y - 0.165, note, fontsize=8.6, color="0.4", style="italic", transform=axa.transAxes)
        y -= 0.29
    axa.text(0, 0.0, "same arithmetic; only the name of\nthe queried variable changes",
             fontsize=8.8, color="0.4", linespacing=1.5, va="bottom", transform=axa.transAxes)

    # ---- (b) headline
    rows = layout_models(sw)
    paired(axb1, sw, rows, "facilitation@v1",
           "(b)  name says the right answer:\n      accuracy gained",
           "points", a.delta, band=True, ylabels=True)
    paired(axb2, sw, rows, "lure_excess@v1",
           "name says a wrong number:\nlure answers above chance",
           "points", a.delta, band=True, ylabels=False)
    h = [plt.Line2D([], [], color=RED, marker="o", ls="", ms=6, label="no chain of thought"),
         plt.Line2D([], [], color=BLUE, marker="s", ls="", ms=6, label="with chain of thought"),
         plt.Rectangle((0, 0), 1, 1, color="0.90", label="within 2 points of zero")]
    axb1.legend(handles=h, loc="upper center", bbox_to_anchor=(1.10, -0.10), ncol=3,
                frameon=False, fontsize=9.5, handletextpad=0.5, columnspacing=1.4)

    # ---- (c) inside the model
    pj = OUT_DIR / "probes" / a.probe_model / f"L{a.level}" / "cot" / f"train_neutral__alltok" / f"{a.role}.json"
    d = json.loads(pj.read_text())
    cond = f"incongruent@{a.role}"
    mte = json.loads((OUT_DIR / "runs" / a.probe_model / f"L{a.level}" / "cot" / f"{cond}__alltok" / "meta.json").read_text())
    layers = sorted({r["layer"] for r in d["results"]}); n_pos = len(mte["pos_labels"])
    t0 = mte["instance_start_token"]; cot0 = mte["n_prompt_tokens"] - t0
    ex = mte["instances"][0]; name = ex["names"][a.role]
    name_tok = [t - t0 for t in ex["name_tokens"][a.role] if t >= t0]

    def curve(metric):
        acc = {}
        for r in d["results"]:
            ev = r["eval"].get(cond)
            if ev and ev.get(metric) is not None:
                acc.setdefault((r["layer"], int(r["position"][1:])), []).append(ev[metric])
        M = np.full((len(layers), n_pos), np.nan)
        for (l, t), v in acc.items():
            M[layers.index(l), t] = float(np.mean(v))
        return np.nanmax(M, axis=0)

    for t in name_tok:
        axc.axvspan(t - 0.5, t + 0.5, color="#E6D9A8", alpha=0.55, lw=0, zorder=0)
    axc.axvline(cot0 - 0.5, color="0.35", lw=1.2, ls="--")
    axc.plot(curve("accuracy"), "-", lw=1.9, color=GREEN, zorder=3,
             label=f"probe reads {ex['values'][a.role]}, the true value of “{name}”")
    axc.plot(curve("lure_rate"), "-", lw=1.9, color=RED, zorder=3,
             label=f"probe reads {ex['lure']}, what the word “{name}” means")
    axc.set_ylim(-0.03, 1.16); axc.set_xlim(-0.5, n_pos - 0.5)
    axc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axc.set_ylabel("share of problems")
    import matplotlib.patches as mpatches
    hc = [plt.Line2D([], [], color=GREEN, lw=1.9, label=f"reads {ex['values'][a.role]}, the true value"),
          plt.Line2D([], [], color=RED, lw=1.9, label=f"reads {ex['lure']}, what “{name}” means"),
          mpatches.Patch(color="#E6D9A8", alpha=0.55, label=f"the name's tokens")]
    axc.legend(handles=hc, loc="upper left", bbox_to_anchor=(0.005, 0.90), frameon=False, fontsize=8.8)
    axc.set_title("(c)  what a probe reads at each token (Llama-3.2-3B)", loc="left")
    axc.set_xticks(range(n_pos))
    axc.set_xticklabels([mte["token_strings"][i].replace("\n", "\\n") for i in range(n_pos)],
                        rotation=90, family="monospace", fontsize=6)
    axc.text(cot0 - 1.0, 1.10, "the problem", ha="right", fontsize=8.8, color="0.3")
    axc.text(cot0 + 0.2, 1.10, "the model's chain of thought", ha="left", fontsize=8.8, color="0.3")
    wt = max((max(v) - t0 for k, v in mte["segment_tokens"].items() if k.endswith(f":value:{a.role}")), default=None)
    if wt is not None:
        axc.axvline(wt, color=GREEN, lw=1.1, ls=":", zorder=2)
        axc.text(wt - 0.8, 1.02, f"the chain writes “{name}={ex['values'][a.role]}”",
                 fontsize=8.8, color="#0F5C40", ha="right", va="center")

    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / f"fig1_master_L{a.level}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    n_models = len({k.split('/')[0] for k in sw})
    print(f"wrote {stem}.png  ({n_models} models at level {a.level})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
