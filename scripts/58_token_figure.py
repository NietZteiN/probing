#!/usr/bin/env python
"""The main representation figure, in the style of Kudo et al.'s Figure 2: the model's actual
token sequence along the x-axis, what the probe reads above it, and the layer sweep below.

    python scripts/58_token_figure.py --model llama32-3b --level 3 --regime cot --role v2

The point of the figure is to answer one question the reader should not have to reconstruct:
*at each token the model has read or written, does its state hold the variable's true value or
the value its name denotes?* So the title names the probe's question, the x-axis prints the real
tokens, a line marks where the model stops reading and starts writing, and the name that carries
the lure is shaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, PROJECT_ROOT  # noqa: E402
from cueconf.display import MODEL, REGIME, rcparams  # noqa: E402

FIG = PROJECT_ROOT / "paper" / "figures"


def grid(d: dict, cond: str, metric: str, n_pos: int, layers: list[int]) -> np.ndarray:
    acc: dict = {}
    for r in d["results"]:
        ev = r["eval"].get(cond)
        if ev is None or ev.get(metric) is None:
            continue
        acc.setdefault((r["layer"], int(r["position"][1:])), []).append(ev[metric])
    M = np.full((len(layers), n_pos), np.nan)
    for (l, t), v in acc.items():
        M[layers.index(l), t] = float(np.mean(v))
    return M


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama32-3b"); ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--regime", default="cot"); ap.add_argument("--role", default="v2")
    ap.add_argument("--train", default="train_neutral")
    a = ap.parse_args()
    plt.rcParams.update(rcparams())

    pj = OUT_DIR / "probes" / a.model / f"L{a.level}" / a.regime / f"{a.train}__alltok" / f"{a.role}.json"
    d = json.loads(pj.read_text())
    cond = f"incongruent@{a.role}"
    mte = json.loads((OUT_DIR / "runs" / a.model / f"L{a.level}" / a.regime / f"{cond}__alltok" / "meta.json").read_text())
    layers = sorted({r["layer"] for r in d["results"]})
    toks = mte["token_strings"]; n_pos = len(mte["pos_labels"])
    t0 = mte["instance_start_token"]
    cot0 = mte["n_prompt_tokens"] - t0
    ex = mte["instances"][0]
    name, true_v, lure_v = ex["names"][a.role], ex["values"][a.role], ex["lure"]
    name_tok = [t - t0 for t in ex["name_tokens"][a.role] if t >= t0]

    A = grid(d, cond, "accuracy", n_pos, layers)     # probe's top-1 == true value
    L = grid(d, cond, "lure_rate", n_pos, layers)    # probe's top-1 == lure value
    acc_curve, lure_curve = np.nanmax(A, axis=0), np.nanmax(L, axis=0)

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.9], hspace=0.06, figure=fig)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for ax in (ax0, ax1):
        for t in name_tok:
            ax.axvspan(t - 0.5, t + 0.5, color="#B5321F", alpha=0.10, lw=0)
        ax.axvline(cot0 - 0.5, color="0.35", lw=1.2, ls="--")
        ax.set_xlim(-0.5, n_pos - 0.5)

    ax0.plot(acc_curve, "-o", ms=3.2, lw=1.5, color="#157A55", label=f"reads the TRUE value ({true_v})")
    ax0.plot(lure_curve, "-o", ms=3.2, lw=1.5, color="#B5321F", label=f"reads the LURE value ({lure_v})")
    ax0.set_ylim(-0.03, 1.05); ax0.set_ylabel("share of problems\n(best layer)")
    ax0.legend(loc="upper left", framealpha=0.95)
    ax0.set_xticks([])
    ax0.text(cot0 / 2, 1.14, "the problem the model reads", ha="center", va="bottom", fontsize=10, color="0.25")
    ax0.text(cot0 + (n_pos - cot0) / 2, 1.14, "the chain of thought the model writes",
             ha="center", va="bottom", fontsize=10, color="0.25")
    ax0.annotate("", xy=(0, 1.11), xytext=(cot0 - 1, 1.11), arrowprops=dict(arrowstyle="<->", color="0.55", lw=0.9))
    ax0.annotate("", xy=(cot0, 1.11), xytext=(n_pos - 1, 1.11), arrowprops=dict(arrowstyle="<->", color="0.55", lw=0.9))

    # annotate the moment the chain writes the variable's value: where the true curve takes over
    write_t = None
    for lab, toks_ in mte["segment_tokens"].items():
        if lab.startswith("cot:") and lab.endswith(f":value:{a.role}"):
            write_t = max(toks_) - t0
    if write_t is not None and 0 <= write_t < n_pos:
        ax0.annotate(f"here the chain writes “{name}={true_v}”",
                     xy=(write_t, acc_curve[write_t]), xytext=(write_t + 1.5, 0.42),
                     fontsize=10, color="#0F5C40",
                     arrowprops=dict(arrowstyle="->", color="#157A55", lw=1.3))

    im = ax1.imshow(A, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                    extent=(-0.5, n_pos - 0.5, layers[0] - 0.5, layers[-1] + 0.5))
    ax1.set_ylabel("layer")
    ax1.set_xticks(range(n_pos))
    ax1.set_xticklabels([t.replace("\n", "\\n") for t in toks], rotation=90, family="monospace", fontsize=7.5)
    ax1.set_xlabel(f"every token of one problem and its solution   (shaded: the name “{name}”, "
                   f"whose value is {true_v} but which denotes {lure_v})", labelpad=6)
    cb = fig.colorbar(im, ax=[ax0, ax1], fraction=0.022, pad=0.012)
    cb.set_label("probe reads the true value (share of problems)")

    fig.suptitle(f"{MODEL.get(a.model, a.model)}, {REGIME.get(a.regime, a.regime)}.   "
                 f"Probe question: what value does the model hold for “{name}”?", fontsize=12, y=0.985)
    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / f"fig2_tokens_{a.model}_L{a.level}_{a.regime}_{a.role}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png  ({n_pos} tokens, chain starts at token {cot0})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
