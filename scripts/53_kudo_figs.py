#!/usr/bin/env python
"""Kudo et al. (2026) figures and tables, reproduced per condition in the cue-conflict setting
(docs/EXPERIMENTS.md §F: E28 and E30). Reads the per-token probes (`--suffix __alltok`).

    python scripts/53_kudo_figs.py --model llama32-3b --level 3 [--regime cot] [--role v2] [--tau 0.9]

Outputs in paper/figures/:
  kudo_fig2_<model>_L<level>_<regime>_<role>.{pdf,png}   accuracy heatmaps (token x layer) for neutral /
        congruent / incongruent with the max-over-layers curve above; lure-rate and margin heatmaps
        for incongruent; x-axis = the tokens of the first instance
  kudo_fig3_<model>_L<level>_<regime>_<role>.{pdf,png}   top-1 probe prediction trajectories (best layer per
        token) on lure-error instances and on random incongruent instances (E30)
and results/summary/<model>/L<level>/<regime>/kudo_table.json: per condition, t*, t*_eq,
Acc<CoT, Acc>CoT (Kudo Tables 2-3) plus t_lure and max lure rate before / after the chain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, PROJECT_ROOT, RESULTS_DIR  # noqa: E402

FIG = PROJECT_ROOT / "paper" / "figures"


def load(model: str, level: int, regime: str, train: str, role: str):
    pj = OUT_DIR / "probes" / model / f"L{level}" / regime / f"{train}__alltok" / f"{role}.json"
    d = json.loads(pj.read_text())
    meta = json.loads((Path(d["train_dir"]) / "meta.json").read_text())
    return d, meta, np.load(pj.with_suffix(".npz"))


def grid(d: dict, cond: str, metric: str, n_pos: int, layers: list[int]) -> np.ndarray:
    """[n_layers, n_pos] mean over seeds of `metric` for condition `cond`."""
    acc = {}
    for r in d["results"]:
        ev = r["eval"].get(cond)
        if ev is None or ev.get(metric) is None:
            continue
        key = (r["layer"], int(r["position"][1:]))
        acc.setdefault(key, []).append(ev[metric])
    M = np.full((len(layers), n_pos), np.nan)
    for (l, t), v in acc.items():
        M[layers.index(l), t] = float(np.mean(v))
    return M


def kudo_table(d: dict, meta: dict, cond: str, tau: float) -> dict:
    layers = sorted({r["layer"] for r in d["results"]})
    n_pos = len(meta["pos_labels"])
    cot0 = meta["n_prompt_tokens"] - meta["instance_start_token"]          # first output token, in instance coordinates
    A = grid(d, cond, "accuracy", n_pos, layers)
    best = np.nanmax(A, axis=0)                                          # max over layers per token
    rel = np.arange(n_pos) - cot0                                        # Kudo's t: 0 at CoT start
    above = [int(rel[t]) for t in range(n_pos) if best[t] > tau]
    t_star = above[0] if above else None
    out = {"t_star": t_star, "acc_pre_cot": float(np.nanmax(best[:cot0])) if cot0 > 0 else None,
           "acc_post_cot": float(np.nanmax(best[cot0:])) if cot0 < n_pos else None}
    # equation index of t*: which segment contains it (Kudo's t*_eq, with input equations negative)
    seg = meta["segment_tokens"]; t0 = meta["instance_start_token"]
    if t_star is not None:
        abs_t = t_star + meta["n_prompt_tokens"]
        for lab, toks in seg.items():
            if abs_t in toks:
                out["t_star_segment"] = lab
    L = grid(d, cond, "lure_rate", n_pos, layers)
    if not np.all(np.isnan(L)):
        lmax = np.nanmax(L, axis=0)
        last = [int(rel[t]) for t in range(n_pos) if lmax[t] > 0.5]
        out.update({"t_lure_last_above_half": (last[-1] if last else None),
                    "lure_rate_max_pre_cot": float(np.nanmax(lmax[:cot0])) if cot0 > 0 else None,
                    "lure_rate_max_post_cot": float(np.nanmax(lmax[cot0:])) if cot0 < n_pos else None})
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--regime", default="cot"); ap.add_argument("--role", default="v2"); ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--train", default="train_neutral"); ap.add_argument("--trajectories", action="store_true")
    a = ap.parse_args()
    d, meta, npz = load(a.model, a.level, a.regime, a.train, a.role)
    layers = sorted({r["layer"] for r in d["results"]})
    n_pos = len(meta["pos_labels"]); toks = meta["token_strings"]
    cot0 = meta["n_prompt_tokens"] - meta["instance_start_token"]
    r = a.role
    conds = ["neutral", f"congruent@{r}", f"incongruent@{r}"]
    FIG.mkdir(parents=True, exist_ok=True)

    # ---- Fig. 2 analogue
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1, 2.6, 2.6]}, sharex=True)
    for j, cond in enumerate(conds):
        A = grid(d, cond, "accuracy", n_pos, layers)
        axes[0, j].plot(np.nanmax(A, axis=0), color="k", lw=1.2, label="acc (max over layers)")
        if "incongruent" in cond:
            Lr = grid(d, cond, "lure_rate", n_pos, layers)
            axes[0, j].plot(np.nanmax(Lr, axis=0), color="#B5321F", lw=1.2, label="lure rate (max)")
            axes[0, j].legend(fontsize=6, loc="upper left")
        axes[0, j].set_ylim(0, 1.02); axes[0, j].axvline(cot0 - 0.5, color="grey", lw=0.8, ls="--")
        axes[0, j].set_title(f"{cond}: value of {r}", fontsize=9)
        im = axes[1, j].imshow(A, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        axes[1, j].axvline(cot0 - 0.5, color="w", lw=0.8, ls="--")
        axes[1, j].set_yticks(range(len(layers))); axes[1, j].set_yticklabels(layers, fontsize=6)
        if "incongruent" in cond:
            Mg = grid(d, cond, "margin_mean", n_pos, layers); v = np.nanmax(np.abs(Mg)) if not np.all(np.isnan(Mg)) else 1
            im2 = axes[2, j].imshow(Mg, aspect="auto", origin="lower", cmap="RdBu", vmin=-v, vmax=v)
            axes[2, j].set_title("margin log p(true) - log p(lure)", fontsize=8)
            fig.colorbar(im2, ax=axes[2, j], fraction=0.03)
        else:
            Lr = grid(d, cond, "accuracy", n_pos, layers)  # placeholder panel: control accuracy if present
            C = np.full_like(A, np.nan)
            for rr in d["results"]:
                if cond in rr.get("control_acc", {}):
                    C[layers.index(rr["layer"]), int(rr["position"][1:])] = rr["control_acc"][cond]
            axes[2, j].imshow(C, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
            axes[2, j].set_title("control-task accuracy (Hewitt & Liang)", fontsize=8)
        axes[2, j].set_yticks(range(len(layers))); axes[2, j].set_yticklabels(layers, fontsize=6)
        axes[2, j].set_xticks(range(n_pos)); axes[2, j].set_xticklabels([t.replace("\n", "⏎") for t in toks], rotation=90, fontsize=5, family="monospace")
        axes[1, j].axvline(cot0 - 0.5, color="w", lw=0.8, ls="--"); axes[2, j].axvline(cot0 - 0.5, color="k", lw=0.8, ls="--")
    fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.015, label="probe accuracy")
    axes[1, 0].set_ylabel("layer"); axes[2, 0].set_ylabel("layer")
    fig.suptitle(f"{a.model}, level {a.level}, {a.regime}: per-token probes for the value of {r} (trained on {a.train})", fontsize=10)
    fig.savefig(FIG / f"kudo_fig2_{a.model}_L{a.level}_{a.regime}_{r}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"kudo_fig2_{a.model}_L{a.level}_{a.regime}_{r}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- Tables 2-3 analogue
    tab = {cond: kudo_table(d, meta, cond, a.tau) for cond in conds + ["letter"] if any(cond in rr["eval"] for rr in d["results"])}
    out = RESULTS_DIR / "summary" / a.model / f"L{a.level}" / a.regime
    out.mkdir(parents=True, exist_ok=True)
    prev = json.loads((out / "kudo_table.json").read_text()) if (out / "kudo_table.json").exists() else {}
    prev[f"{a.train}/{r}"] = tab
    (out / "kudo_table.json").write_text(json.dumps(prev, indent=1))
    print(json.dumps(tab, indent=1))

    # ---- Fig. 3 analogue: trajectories (E30)
    if a.trajectories:
        cond = f"incongruent@{r}"
        ids = d["test_ids"][cond]
        # best layer per token by neutral accuracy
        A = grid(d, "neutral", "accuracy", n_pos, layers)
        best_layer = [layers[int(np.nanargmax(A[:, t]))] for t in range(n_pos)]
        beh = {}
        bf = OUT_DIR / "runs" / a.model / f"L{a.level}" / a.regime / cond / "behavior.jsonl"
        if bf.exists():
            beh = {json.loads(l)["id"]: json.loads(l) for l in bf.open()}
        lure_err = [i for i, x in enumerate(ids) if beh.get(x, {}).get("pred_is_lure")]
        rng = np.random.default_rng(0)
        sample = list(lure_err[:12]) + list(rng.choice([i for i in range(len(ids)) if i not in lure_err], 12, replace=False))
        fig, ax = plt.subplots(figsize=(11, 0.35 * len(sample) + 1.5))
        meta_te = json.loads((OUT_DIR / "runs" / a.model / f"L{a.level}" / a.regime / f"{cond}__alltok" / "meta.json").read_text())
        for row, i in enumerate(sample):
            inst = meta_te["instances"][i]; true = inst["values"][r]; lure = inst["lure"]
            for t in range(n_pos):
                key = f"t{t}/L{best_layer[t]}/s0/{cond}/pred"
                if key not in npz.files:
                    continue
                p = int(npz[key][i])
                col = "#157A55" if p == true else ("#B5321F" if p == lure else "#C8CCD3")
                ax.scatter(t, row, s=14, color=col, marker="s")
            ax.text(-1, row, ("lure err " if i in lure_err else "ok ") + inst["id"].split("-")[-2], fontsize=5, ha="right", va="center")
        ax.axvline(cot0 - 0.5, color="grey", lw=0.8, ls="--")
        ax.set_xticks(range(n_pos)); ax.set_xticklabels([t.replace("\n", "⏎") for t in toks], rotation=90, fontsize=5, family="monospace")
        ax.set_yticks([]); ax.set_xlim(-0.5, n_pos - 0.5)
        ax.set_title(f"top-1 probe prediction for {r} at the best layer per token: green = true, red = lure, grey = other", fontsize=8)
        fig.savefig(FIG / f"kudo_fig3_{a.model}_L{a.level}_{a.regime}_{r}.pdf", bbox_inches="tight")
        fig.savefig(FIG / f"kudo_fig3_{a.model}_L{a.level}_{a.regime}_{r}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"figures -> {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
