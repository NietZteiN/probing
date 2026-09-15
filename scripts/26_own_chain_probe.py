#!/usr/bin/env python
"""Probe the model's OWN chain on problems it gets wrong (one GPU job).

The cached hidden states used by every other figure come from a forced pass over the GOLD chain,
so they cannot show what the model held while it was making its own mistake. This script:
  1. trains the per-token probes from the existing neutral training cache and KEEPS the weights;
  2. re-runs the model over prompt + its own stored generation for chosen instances;
  3. applies the probes at every token of that sequence;
  4. writes results/summary/<model>/L<level>/<regime>/own_chain_<role>.json for plotting.

    python scripts/26_own_chain_probe.py --model llama32-3b --level 3 --regime cot --role v1 --n 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, RESULTS_DIR, model_entry  # noqa: E402
from cueconf.probes import BatchedProbes  # noqa: E402
from cueconf.prompts import build_prompt, make_demos, parse_regime, scheme_of, demo_seed_of  # noqa: E402
from cueconf.runner import load_model, text_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama32-3b"); ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--regime", default="cot"); ap.add_argument("--role", default="v1")
    ap.add_argument("--n", type=int, default=6, help="wrong instances to probe")
    ap.add_argument("--layer-stride", type=int, default=2); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    cond = f"incongruent@{a.role}"
    base = OUT_DIR / "runs" / a.model / f"L{a.level}" / a.regime

    # ---- 1. train probes from the cached neutral training states, at the labelled positions
    tr = base / "train_neutral"
    mtr = json.loads((tr / "meta.json").read_text())
    H = np.load(tr / "hidden.npy", mmap_mode="r")            # [n, n_pos, n_layers, d]
    layers = list(range(0, H.shape[2], a.layer_stride))
    y = np.array([x["values"][a.role] for x in mtr["instances"]], dtype=np.int64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # the value step is the position the paper reports; train there, one probe per (layer, seed)
    pos = mtr["pos_labels"].index(f"cotpre@{a.role}") if f"cotpre@{a.role}" in mtr["pos_labels"] else \
        mtr["pos_labels"].index("anspre")
    X = torch.tensor(np.asarray(H[:, pos][:, layers, :], dtype=np.float32), device=dev)   # [n, L, d]
    K = len(layers) * len(a.seeds)
    Xk = X.permute(1, 0, 2).repeat_interleave(len(a.seeds), 0).contiguous()               # [K, n, d]
    probes = BatchedProbes(K, X.shape[-1], list(a.seeds), device=dev)
    probes.fit_sgd(Xk, torch.tensor(y, device=dev))
    print(f"trained {K} probes at position {mtr['pos_labels'][pos]} on {len(y)} instances", flush=True)

    # ---- 2. the model over its own generation
    rows = [json.loads(l) for l in (base / cond / "behavior.jsonl").open()]
    inst = {x["id"]: x for x in json.loads((base / f"{cond}__alltok" / "meta.json").read_text())["instances"]} \
        if (base / f"{cond}__alltok" / "meta.json").exists() else {}
    wrong = [r for r in rows if not r["correct"]][: a.n]
    m = model_entry(a.model); tok, model = load_model(m["hf_id"])
    demos = make_demos(a.level, scheme_of("incongruent"), n=parse_regime(a.regime)[1], seed=demo_seed_of(a.regime))
    out = []
    for r in wrong:
        x = inst.get(r["id"])
        names = x["names"] if x else r.get("names")
        if names is None:
            continue
        from cueconf.generator import Instance
        xi = next(i for i in __import__("cueconf.generator", fromlist=["read_jsonl"]).read_jsonl(
            Path(__file__).resolve().parents[1] / "data" / f"L{a.level}" / "test_sets.jsonl") if i.id == r["id"])
        prompt = build_prompt(xi, demos, a.regime)
        full = prompt + r["generation"]
        enc = tok(full, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
        n_prompt = len(tok(prompt, add_special_tokens=True)["input_ids"])
        with torch.no_grad():
            hs = model(input_ids=enc["input_ids"].to(model.device), output_hidden_states=True).hidden_states
        Hf = torch.stack([hs[l][0] for l in layers], dim=1).float()          # [T, L, d]
        T = Hf.shape[0]
        preds = np.zeros((T, len(layers)), dtype=np.int64)
        for li in range(len(layers)):
            lp = torch.stack([probes.probe(li * len(a.seeds) + s, 0).logprobs(Hf[:, li]) for s in range(len(a.seeds))]).mean(0)
            preds[:, li] = lp.argmax(-1).cpu().numpy()
        out.append({"id": r["id"], "answer": r["answer"], "lure": r["lure"], "pred": r["pred"],
                    "true_value": xi.values[a.role], "name": names[a.role], "n_prompt_tokens": n_prompt,
                    "tokens": [tok.decode([t]) for t in enc["input_ids"][0].tolist()],
                    "layers": layers, "probe_pred": preds.tolist(),
                    "generation": r["generation"]})
        print(f"  {r['id']}: answered {r['pred']} (true {r['answer']}, lure {r['lure']}), {T} tokens", flush=True)
    p = RESULTS_DIR / "summary" / a.model / f"L{a.level}" / a.regime
    p.mkdir(parents=True, exist_ok=True)
    (p / f"own_chain_{a.role}.json").write_text(json.dumps(out))
    print(f"wrote {p / f'own_chain_{a.role}.json'} ({len(out)} instances)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
