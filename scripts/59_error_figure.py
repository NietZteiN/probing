#!/usr/bin/env python
"""Figure 2's question asked separately of the problems the model got right and the problems it
got wrong: at each token, does the state hold the variable's true value, the value its name
denotes, or the value the model will actually answer?

IMPORTANT. The cached states come from a FORCED pass over the gold chain (`runner.forced_pass`),
not from the model's own generation. The split is by what the model answered when generating
freely, so the lower panel reads: "on the problems this model gets wrong unaided, what does its
state hold when the correct chain is supplied?" It is not a picture of the model's own mistake.
For that, see `26_own_chain_probe.py`, which re-runs the model over its own generated chain.

    python scripts/59_error_figure.py --model llama32-3b --level 3 --regime cot --role v1

The layer is chosen per token on the NEUTRAL condition, which shares no instances with the split
being plotted, so the choice cannot be tuned to the outcome. Curves are means over three probe
seeds. `n` for each panel is printed in its title, because the wrong-answer panel is a small
subset wherever the model is accurate.
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
GREEN, RED, PURPLE = "#157A55", "#B5321F", "#6A3D9A"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama32-3b"); ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--regime", default="cot"); ap.add_argument("--role", default="v1")
    ap.add_argument("--train", default="train_neutral")
    a = ap.parse_args()
    plt.rcParams.update(rcparams())
    cond = f"incongruent@{a.role}"
    pdir = OUT_DIR / "probes" / a.model / f"L{a.level}" / a.regime / f"{a.train}__alltok"
    d = json.loads((pdir / f"{a.role}.json").read_text())
    z = np.load(pdir / f"{a.role}.npz")
    mte = json.loads((OUT_DIR / "runs" / a.model / f"L{a.level}" / a.regime / f"{cond}__alltok" / "meta.json").read_text())
    n_pos = len(mte["pos_labels"]); t0 = mte["instance_start_token"]; cot0 = mte["n_prompt_tokens"] - t0
    layers = sorted({int(k.split("/")[1][1:]) for k in z.files}); seeds = sorted({k.split("/")[2] for k in z.files})

# behaviour (outcome) and meta (names, values), aligned to the probe's instance order; runs
    # written before 2026-09-14 have no `names` in behavior.jsonl, so those come from meta
    rows = {r["id"]: r for r in (json.loads(l) for l in
            (OUT_DIR / "runs" / a.model / f"L{a.level}" / a.regime / cond / "behavior.jsonl").open())}
    inst = {x["id"]: x for x in mte["instances"]}
    ids = [i for i in d["test_ids"][cond] if i in rows and i in inst]
    keep = np.array([i in rows and i in inst for i in d["test_ids"][cond]])
    correct = np.array([bool(rows[i]["correct"]) for i in ids])
    true_v = np.array([inst[i]["values"][a.role] for i in ids])
    lure_v = np.array([inst[i]["lure"] if inst[i]["lure"] is not None else -1 for i in ids])
    pred_v = np.array([rows[i]["pred"] if rows[i]["pred"] is not None else -1 for i in ids])
    name_tokens = [t - t0 for t in inst[ids[0]]["name_tokens"][a.role] if t >= t0]
    name = inst[ids[0]]["names"][a.role]

    def stack(condition: str, field: str, mask=None) -> np.ndarray:
        """[n_pos, n_layers, n_inst] averaged over seeds."""
        n_inst = len(z[f"t0/L{layers[0]}/{seeds[0]}/{condition}/{field}"])
        out = np.full((n_pos, len(layers), n_inst), np.nan, dtype=np.float32)
        for t in range(n_pos):
            for li, L in enumerate(layers):
                vals = [z[f"t{t}/L{L}/{s}/{condition}/{field}"] for s in seeds
                        if f"t{t}/L{L}/{s}/{condition}/{field}" in z.files]
                if vals:
                    out[t, li] = np.mean(vals, axis=0)
        return out[:, :, mask] if mask is not None else out

    # layer choice per token from the neutral condition: independent of the correct/wrong split
    neu_meta = json.loads((OUT_DIR / "runs" / a.model / f"L{a.level}" / a.regime / "neutral__alltok" / "meta.json").read_text())
    neu_inst = {x["id"]: x for x in neu_meta["instances"]}
    neu_true = np.array([neu_inst[i]["values"][a.role] for i in d["test_ids"]["neutral"]])
    neu_pred = stack("neutral", "pred")
    best = np.nanargmax(np.nanmean(neu_pred == neu_true[None, None, :], axis=2), axis=1)

    pred = stack(cond, "pred", keep)                # [n_pos, n_layers, n_kept]
    sel = np.stack([pred[t, best[t]] for t in range(n_pos)])   # [n_pos, n_inst]

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.22})
    for ax, mask, title in ((axes[0], correct, "problems the model answers correctly unaided"),
                            (axes[1], ~correct, "problems the model answers wrongly unaided")):
        n = int(mask.sum())
        for t_ in name_tokens:
            ax.axvspan(t_ - 0.5, t_ + 0.5, color="#E6D9A8", alpha=0.55, lw=0, zorder=0)
        ax.axvline(cot0 - 0.5, color="0.35", lw=1.2, ls="--", zorder=1)
        if n:
            ax.plot((sel[:, mask] == true_v[mask]).mean(1), "-", lw=1.9, color=GREEN, zorder=3,
                    label="probe reads the true value")
            ax.plot((sel[:, mask] == lure_v[mask]).mean(1), "-", lw=1.9, color=RED, zorder=3,
                    label="probe reads the value the name denotes")
            if not mask.all():
                ax.plot((sel[:, mask] == pred_v[mask]).mean(1), "-", lw=1.9, color=PURPLE, zorder=3,
                        label="probe reads the answer the model gives")
        ax.set_ylim(-0.03, 1.05); ax.set_xlim(-0.5, n_pos - 0.5)
        ax.set_ylabel("share of problems")
        ax.set_title(f"{title}  (n = {n:,})", loc="left", fontsize=10.5)
        ax.legend(loc="upper left", frameon=False, fontsize=8.6)
    axes[1].set_xticks(range(n_pos))
    axes[1].set_xticklabels([mte["token_strings"][i].replace("\n", "\\n") for i in range(n_pos)],
                            rotation=90, family="monospace", fontsize=6)
    axes[0].text(cot0 - 1.0, 1.0, "the problem", ha="right", fontsize=8.8, color="0.3")
    axes[0].text(cot0 + 0.2, 1.0, "the model's chain of thought", ha="left", fontsize=8.8, color="0.3")
    fig.suptitle(f"What the state holds for “{name}” with the correct chain supplied, split by whether "
                 f"the model answers that problem correctly on its own\n"
                 f"({MODEL.get(a.model, a.model)}, level {a.level}, {REGIME.get(a.regime, a.regime)}; "
                 f"in the upper panel the true value and the model's answer coincide)",
                 x=0.005, ha="left", fontsize=10.5)
    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / f"fig4_errors_{a.model}_L{a.level}_{a.regime}_{a.role}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {stem}.png  ({int(correct.sum())} correct, {int((~correct).sum())} wrong, "
          f"layers chosen on neutral)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
