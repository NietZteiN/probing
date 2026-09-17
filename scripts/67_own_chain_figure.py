#!/usr/bin/env python
"""Probe predictions over the model's OWN generated chain, for the instances it answered wrongly.

Input: results/summary/<model>/L<level>/<regime>/own_chain_<role>.json from 26_own_chain_probe.py.
Probes are trained on neutral FORCED-pass states at the target's value step (cotpre@<role>),
one per layer, three seeds averaged, then applied to every generated token of the model's own
chain. Unlike 59_error_figure.py and 64_kudo_fig3.py, nothing here is a forced gold chain: the
states are the model's own mistake. Only eight instances exist, so this is a picture, not a rate.

    python scripts/67_own_chain_figure.py --model llama32-3b --level 3 --regime cot --role v1

Colour code as in 64_kudo_fig3.py: green the variable's true value, red the value the model
answered, amber the value its name denotes, grey anything else. Boxed columns are the name's own
tokens inside the generation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from cueconf.display import MODEL, REGIME, rcparams  # noqa: E402

FIG = PROJECT_ROOT / "paper" / "figures"
GREEN, RED, AMBER, GREY = "#157A55", "#B5321F", "#D9A21B", "#D9D9D9"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mc
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama32-3b"); ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--regime", default="cot"); ap.add_argument("--role", default="v1")
    ap.add_argument("--max-instances", type=int, default=8)
    a = ap.parse_args()
    src = RESULTS_DIR / "summary" / a.model / f"L{a.level}" / a.regime / f"own_chain_{a.role}.json"
    rows = json.loads(src.read_text())[: a.max_instances]
    assert rows, f"nothing in {src}"
    plt.rcParams.update(rcparams())
    ncol = 2; nrow = (len(rows) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.6, 2.15 * nrow), squeeze=False)
    for ax in axes.ravel()[len(rows):]: ax.axis("off")
    for ax, r in zip(axes.ravel(), rows):
        layers = r["layers"]; t0 = r["n_prompt_tokens"]
        P = np.array(r["probe_pred"], dtype=int)[t0:].T                      # [L, T_gen]
        toks = r["tokens"][t0:]; assert P.shape[1] == len(toks)
        true_v, lure_v, pred_v = r["true_value"], r["lure"], r["pred"]
        wrong = pred_v != r["answer"]
        C = np.zeros(P.shape + (3,))
        for li in range(P.shape[0]):
            for t in range(P.shape[1]):
                v = P[li, t]
                col = GREEN if v == true_v else (RED if (wrong and v == pred_v) else (AMBER if v == lure_v else GREY))
                C[li, t] = mc.to_rgb(col)
        ax.imshow(C, aspect="auto", origin="lower", interpolation="nearest")
        for li in range(P.shape[0]):
            for t in range(P.shape[1]):
                strong = P[li, t] in (true_v, lure_v) or (wrong and P[li, t] == pred_v)
                ax.text(t, li, str(P[li, t]), ha="center", va="center", fontsize=5.2, color="white" if strong else "0.35")
        for t, s in enumerate(toks):                                        # the name's own tokens
            if s.strip() == r["name"]:
                ax.add_patch(plt.Rectangle((t - 0.5, -0.5), 1, P.shape[0], fill=False, ec="k", lw=1.0, zorder=5))
        step = max(1, len(layers) // 5)
        ax.set_yticks(range(0, len(layers), step)); ax.set_yticklabels([layers[i] for i in range(0, len(layers), step)], fontsize=7)
        ax.set_ylabel("layer", fontsize=8)
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels([s.replace("\n", "\\n") for s in toks], rotation=90, family="monospace", fontsize=5.4)
        ax.tick_params(axis="x", length=0, pad=1)
        ok = "correct" if not wrong else f"answered {pred_v}"
        ax.set_title(f"“{r['name']}” holds {true_v}; the name denotes {lure_v}; model {ok}", loc="left", fontsize=8)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
         ((GREEN, "true value"), (RED, "the model's answer (when wrong)"), (AMBER, "what the name denotes"), (GREY, "other"))]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"Top-1 probe prediction over the model's own chain "
                 f"({MODEL.get(a.model, a.model)}, L{a.level}, {REGIME.get(a.regime, a.regime)}; probes trained at the "
                 f"value step of “{a.role}” on neutral problems; {len(rows)} instances)", y=1.03, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / f"fig_own_chain_{a.model}_L{a.level}_{a.regime}_{a.role}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.1)
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight", pad_inches=0.1)
    print("wrote", stem.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
