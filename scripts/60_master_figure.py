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


def forest(ax, sw: dict, contrast: str, title: str, xlabel: str, delta: float, show_legend: bool,
           band: bool = True):
    models = sorted({k.split("/")[0] for k in sw}, key=lambda m: (MODEL.get(m, m)))
    ys = np.arange(len(models))
    if band:
        ax.axvspan(-100 * delta, 100 * delta, color="0.88", zorder=0)
        ax.text(0, ax.get_ylim()[1], "", ha="center")
    ax.axvline(0, color="0.4", lw=0.9, zorder=1)
    for regime, colour, marker, off, lab in (("direct", RED, "o", +0.17, "no chain of thought"),
                                             ("cot", BLUE, "s", -0.17, "with chain of thought")):
        xs, los, his, yy = [], [], [], []
        for i, m in enumerate(models):
            c = sw.get(f"{m}/{regime}", {}).get("contrasts", {}).get(contrast)
            if not c:
                continue
            xs.append(100 * c["pooled_mean"]); los.append(100 * c["ci95"][0]); his.append(100 * c["ci95"][1])
            yy.append(i + off)
        if xs:
            ax.errorbar(xs, yy, xerr=[np.array(xs) - np.array(los), np.array(his) - np.array(xs)],
                        fmt=marker, ms=5, color=colour, ecolor=colour, elinewidth=1.2, capsize=2,
                        label=lab, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([MODEL.get(m, m) for m in models])
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")
    ax.grid(axis="x", color="0.92", lw=0.7, zorder=0)
    if show_legend:
        ax.legend(loc="lower right", framealpha=0.95)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3); ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--probe-model", default="llama32-3b"); ap.add_argument("--role", default="v2")
    a = ap.parse_args()
    plt.rcParams.update(rcparams())
    sw = json.loads((RESULTS_DIR / "summary" / f"seed_sweep_L{a.level}.json").read_text())

    fig = plt.figure(figsize=(14.6, 4.8))
    gs = GridSpec(2, 3, width_ratios=[0.95, 1.25, 1.7], height_ratios=[1, 1], wspace=0.60, hspace=0.62, figure=fig)
    axa = fig.add_subplot(gs[:, 0]); axa.axis("off")
    axb1, axb2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    axc = fig.add_subplot(gs[:, 2])

    # ---- (a) the manipulation
    axa.set_title("(a)  the same arithmetic, three names", loc="left", fontsize=11)
    rows = [("neutral", "pen=1 + cup, cup=2 + 3; pen=?", GREY, None),
            ("the name agrees", "pen=1 + five, five=2 + 3; pen=?", GREEN, "five really is 5"),
            ("the name lies", "pen=1 + two, two=2 + 3; pen=?", RED, "two is really 5")]
    y = 0.92
    for lab, prob, col, note in rows:
        axa.text(0, y, lab, fontsize=10, color=col, weight="bold", transform=axa.transAxes)
        axa.text(0, y - 0.085, prob, fontsize=9, family="monospace", color=col, transform=axa.transAxes)
        if note:
            axa.text(0, y - 0.155, note, fontsize=8.5, color="0.4", style="italic", transform=axa.transAxes)
            y -= 0.27
        else:
            y -= 0.20
    axa.text(0, 0.21, "The answer is 6 in all three.", fontsize=9, color="0.2",
             va="top", transform=axa.transAxes)
    axa.text(0, 0.14, "The model either writes the steps\n"
                      "\u201ccup=2 + 3, cup=5, pen=1 + cup,\n pen=1 + 5, pen=6\u201d,\n"
                      "or answers \u201cpen=6\u201d directly.",
             fontsize=8.6, color="0.4", linespacing=1.6, va="top", transform=axa.transAxes)

    # ---- (b) headline
    forest(axb1, sw, "facilitation@v1", "(b)  when the name agrees, does it help?",
           "accuracy gained (points)", a.delta, show_legend=False, band=False)
    forest(axb2, sw, "lure_excess@v1", "when the name lies, is it answered?",
           "extra answers equal to the name (points)", a.delta, show_legend=True)

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
        axc.axvspan(t - 0.5, t + 0.5, color=RED, alpha=0.10, lw=0)
    axc.axvline(cot0 - 0.5, color="0.35", lw=1.2, ls="--")
    axc.plot(curve("accuracy"), "-", lw=1.8, color=GREEN, label=f"probe reads the true value of “{name}”")
    axc.plot(curve("lure_rate"), "-", lw=1.8, color=RED, label=f"probe reads what “{name}” denotes")
    axc.set_ylim(-0.03, 1.22); axc.set_xlim(-0.5, n_pos - 0.5)
    axc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axc.set_ylabel("share of problems")
    axc.legend(loc="lower left", bbox_to_anchor=(0.005, 0.02), framealpha=0.95, fontsize=8.5)
    axc.set_title("(c)  what the model holds, token by token", loc="left")
    axc.set_xticks(range(n_pos))
    axc.set_xticklabels([mte["token_strings"][i].replace("\n", "\\n") for i in range(n_pos)],
                        rotation=90, family="monospace", fontsize=6)
    axc.text(cot0 / 2, 1.14, "the problem", ha="center", fontsize=9, color="0.3")
    axc.text(cot0 + (n_pos - cot0) / 2, 1.14, "the model's chain of thought", ha="center", fontsize=9, color="0.3")
    wt = max((max(v) - t0 for k, v in mte["segment_tokens"].items() if k.endswith(f":value:{a.role}")), default=None)
    if wt is not None:
        axc.axvline(wt, color=GREEN, lw=1.1, ls=":", zorder=2)
        axc.text(wt - 0.6, 1.04, f"the chain writes “{name}={ex['values'][a.role]}”",
                 fontsize=9, color="#0F5C40", ha="right", va="center")

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
