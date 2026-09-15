#!/usr/bin/env python
"""Kudo et al. Figure 3 analogue: for one problem, the probe's top-1 prediction in every
(layer, token) cell, with the digit printed and coloured by what it is.

    python scripts/64_kudo_fig3.py --model llama32-3b --level 3 --regime cot --role v1 [--errors]

Kudo colour the gold label green and the value the model wrongly generated red. A cue conflict
adds a third value worth seeing, so the key is:
    green  the variable's true value
    red    the value the model actually answered (shown only when that answer is wrong)
    amber  the value the variable's NAME denotes (the lure), when it is neither of those
    grey   any other digit
`--errors` picks problems the model answered wrongly; otherwise the first problems in the group.
States come from the forced pass over the gold chain; `26_own_chain_probe.py` gives the same
picture over the model's own generated chain.
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
GREEN, RED, AMBER, GREY = "#157A55", "#B5321F", "#C98A0B", "#B8BCC2"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama32-3b"); ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--regime", default="cot"); ap.add_argument("--role", default="v1")
    ap.add_argument("--train", default="train_neutral"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--errors", action="store_true"); ap.add_argument("--lure-errors", action="store_true")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="skip per-instance token text (then the axis shows the FIRST instance's tokens, which "
                         "differ from the plotted instance's names and digits)")
    a = ap.parse_args()
    plt.rcParams.update(rcparams())
    cond = f"incongruent@{a.role}"
    pdir = OUT_DIR / "probes" / a.model / f"L{a.level}" / a.regime / f"{a.train}__alltok"
    d = json.loads((pdir / f"{a.role}.json").read_text()); z = np.load(pdir / f"{a.role}.npz")
    base = OUT_DIR / "runs" / a.model / f"L{a.level}" / a.regime
    mte = json.loads((base / f"{cond}__alltok" / "meta.json").read_text())
    inst = {x["id"]: x for x in mte["instances"]}
    rows = {r["id"]: r for r in (json.loads(l) for l in (base / cond / "behavior.jsonl").open())}
    ids = d["test_ids"][cond]
    layers = sorted({int(k.split("/")[1][1:]) for k in z.files})
    seeds = sorted({k.split("/")[2] for k in z.files})
    n_pos = len(mte["pos_labels"]); t0 = mte["instance_start_token"]; cot0 = mte["n_prompt_tokens"] - t0

    pick = [i for i in ids if i in rows and i in inst]
    if a.lure_errors:
        pick = [i for i in pick if rows[i]["pred_is_lure"]]
    elif a.errors:
        pick = [i for i in pick if not rows[i]["correct"]]
    pick = pick[: a.n]
    if not pick:
        print("no instances match the filter"); return 1
    idx = {i: k for k, i in enumerate(ids)}

    # the axis must show the PLOTTED instance's own tokens: meta stores only the first instance's,
    # so re-tokenize each one from the dataset (same layout, different names and digits)
    per_tokens = {}
    if not a.no_tokenizer:
        from transformers import AutoTokenizer
        from cueconf.config import DATA_DIR, model_entry
        from cueconf.generator import read_jsonl
        from cueconf.prompts import build_prompt, demo_seed_of, make_demos, parse_regime, scheme_of, layout, tokenize_layout
        tok = AutoTokenizer.from_pretrained(model_entry(a.model)["hf_id"])
        dset = {x.id: x for x in read_jsonl(DATA_DIR / f"L{a.level}" / "test_sets.jsonl")}
        dem = make_demos(a.level, scheme_of("incongruent"), n=parse_regime(a.regime)[1], seed=demo_seed_of(a.regime))
        for iid in pick:
            lay = layout(dset[iid], dem, a.regime)
            tt = tokenize_layout(tok, lay)
            per_tokens[iid] = tt["token_strings"][t0: t0 + n_pos]

    fig, axes = plt.subplots(len(pick) + 1, 1, figsize=(14.5, 3.4 * (len(pick) + 1) + 1.2), squeeze=False)
    fig.subplots_adjust(hspace=0.55, top=0.93)
    # --- baseline panel: the probe's modal prediction over NEUTRAL problems. Where the probe has
    # nothing to read it still emits a class (at layer 0 it says "5" for every instance), so a cell
    # only carries information where it differs from this.
    axb = axes[0, 0]
    B = np.zeros((len(layers), n_pos), dtype=int)
    for li, L in enumerate(layers):
        for t in range(n_pos):
            k_ = f"t{t}/L{L}/{seeds[0]}/neutral/pred"
            B[li, t] = np.bincount(z[k_].astype(int)).argmax() if k_ in z.files else -1
    axb.imshow(np.zeros(B.shape + (3,)) + 0.93, aspect="auto", origin="lower", interpolation="nearest")
    for li in range(B.shape[0]):
        for t in range(B.shape[1]):
            axb.text(t, li, str(B[li, t]), ha="center", va="center", fontsize=5.2, color="0.35")
    axb.axvline(cot0 - 0.5, color="k", lw=1.3, ls="--")
    axb.set_yticks(range(0, len(layers), max(1, len(layers) // 8)))
    axb.set_yticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers) // 8))], fontsize=7)
    axb.set_ylabel("layer"); axb.set_xticks([]); 
    axb.set_title("BASELINE: the probe's most common prediction on neutral problems (its default where it "
                  "reads nothing) — a coloured cell below means something only where it differs from this",
                  loc="left", fontsize=9.5)

    for ax, iid in zip(axes[1:, 0], pick):
        k = idx[iid]; x = inst[iid]; r = rows[iid]
        true_v, lure_v, pred_v = x["values"][a.role], x["lure"], r["pred"]
        # majority vote over probe seeds, per (token, layer)
        P = np.zeros((len(layers), n_pos), dtype=int)
        for li, L in enumerate(layers):
            for t in range(n_pos):
                vals = [int(z[f"t{t}/L{L}/{s}/{cond}/pred"][k]) for s in seeds
                        if f"t{t}/L{L}/{s}/{cond}/pred" in z.files]
                P[li, t] = np.bincount(vals).argmax() if vals else -1
        C = np.zeros(P.shape + (3,))
        import matplotlib.colors as mc
        for li in range(P.shape[0]):
            for t in range(P.shape[1]):
                v = P[li, t]
                col = GREEN if v == true_v else (RED if (pred_v is not None and v == pred_v and not r["correct"])
                                                 else (AMBER if v == lure_v else GREY))
                C[li, t] = mc.to_rgb(col)
        ax.imshow(C, aspect="auto", origin="lower", interpolation="nearest")
        for li in range(P.shape[0]):
            for t in range(P.shape[1]):
                ax.text(t, li, str(P[li, t]), ha="center", va="center", fontsize=5.2,
                        color="white" if P[li, t] in (true_v, lure_v) or (pred_v is not None and P[li, t] == pred_v and not r["correct"]) else "0.35")
        ax.axvline(cot0 - 0.5, color="k", lw=1.3, ls="--")
        for t_ in [t - t0 for t in x["name_tokens"][a.role] if t >= t0]:
            ax.add_patch(plt.Rectangle((t_ - 0.5, -0.5), 1, P.shape[0], fill=False, ec="k", lw=1.1, zorder=5))
        ax.set_yticks(range(0, len(layers), max(1, len(layers) // 8)))
        ax.set_yticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers) // 8))], fontsize=7)
        ax.set_ylabel("layer")
        ok = "correct" if r["correct"] else f"WRONG: answered {pred_v}"
        ax.set_title(f"“{x['names'][a.role]}” holds {true_v}; the name denotes {lure_v}; model {ok}",
                     loc="left", fontsize=9.5)
        toks_i = list(per_tokens.get(iid, mte["token_strings"][:n_pos]))[:n_pos]
        ax.set_xticks(range(len(toks_i)))
        ax.set_xticklabels([t.replace("\n", "\\n") for t in toks_i], rotation=90,
                           family="monospace", fontsize=5.4)
        ax.tick_params(axis="x", length=0, pad=1)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
         ((GREEN, "the variable's true value"), (RED, "the value the model answered (when wrong)"),
          (AMBER, "the value the name denotes"), (GREY, "any other digit"))]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.02, 0.975), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Top-1 probe prediction at every layer and token "
                 f"({MODEL.get(a.model, a.model)}, level {a.level}, {REGIME.get(a.regime, a.regime)}; "
                 f"boxed columns are the name's tokens, dashed line starts the chain)",
                 x=0.005, y=0.995, ha="left", fontsize=10.5)
    FIG.mkdir(parents=True, exist_ok=True)
    tag = "lureerr" if a.lure_errors else ("err" if a.errors else "ok")
    stem = FIG / f"fig5_kudo3_{a.model}_L{a.level}_{a.regime}_{a.role}_{tag}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.35)
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight", pad_inches=0.35)
    print(f"wrote {stem}.png  ({len(pick)} instances: {pick})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
