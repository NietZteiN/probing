#!/usr/bin/env python
"""Measure before a value in saved generated chains, with neutral transfer calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cueconf.config import DATA_DIR, OUT_DIR, RESULTS_DIR, model_entry
from cueconf.followups import fixed_sample, value_boundary
from cueconf.generator import read_jsonl
from cueconf.probes import BatchedProbes, control_labels
from cueconf.prompts import assert_before_value, build_prompt, make_demos
from cueconf.runner import load_model


def records(path):
    return [json.loads(line) for line in path.open()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    args = ap.parse_args()
    instances = {x.id: x for x in read_jsonl(DATA_DIR / "L3/test_sets.jsonl")}
    sample = fixed_sample(x.set_id for x in instances.values())
    root = OUT_DIR / "round5_generated" / args.model
    root.mkdir(parents=True, exist_ok=True)
    tok, model = load_model(model_entry(args.model)["hf_id"])
    for role in ("v1", "v2"):
        source = OUT_DIR / "probes" / args.model / "L3/cot/train_neutral" / f"{role}.json"
        original = json.loads(source.read_text())
        candidates = {}
        for row in original["results"]:
            if row["position"] == f"cotpre@{role}":
                candidates.setdefault(row["layer"], []).append(row)
        layer = max(candidates, key=lambda l: np.mean([r["eval"]["neutral"]["accuracy"] -
            r["control_acc"]["neutral"] for r in candidates[l]]))
        train_dir = Path(original["train_dir"])
        train = json.loads((train_dir / "meta.json").read_text())
        if train["n"] != 10000:
            raise ValueError("expected the 10000-row independent neutral training cache")
        signature = hashlib.sha256((source.read_text() + (train_dir / "meta.json").read_text()).encode()).hexdigest()
        weights_path = root / f"{role}_weights.npz"
        weights_meta = root / f"{role}_weights.json"
        if weights_path.exists() and weights_meta.exists():
            if json.loads(weights_meta.read_text())["signature"] != signature:
                raise ValueError("existing weights belong to different probe inputs")
            with np.load(weights_path) as stored:
                W, b, cW, cb = (torch.tensor(stored[key], device="cuda") for key in ("W", "b", "cW", "cb"))
        else:
            hidden = np.load(train_dir / "hidden.npy", mmap_mode="r")
            pi, li = train["pos_labels"].index(f"cotpre@{role}"), train["layers"].index(layer)
            X = torch.tensor(np.asarray(hidden[:, pi, li], dtype=np.float32), device="cuda")
            Xk = X.unsqueeze(0).expand(3, -1, -1).contiguous()
            labels = torch.tensor([x["values"][role] for x in train["instances"]], device="cuda")
            probes = BatchedProbes(3, X.shape[-1], [0, 1, 2], "cuda")
            probes.fit_sgd(Xk, labels)
            controls = BatchedProbes(3, X.shape[-1], [0, 1, 2], "cuda")
            controls.fit_sgd(Xk, torch.tensor(control_labels(train, role), device="cuda"))
            W, b, cW, cb = probes.W, probes.b, controls.W, controls.b
            np.savez_compressed(weights_path, W=W.cpu().numpy(), b=b.cpu().numpy(),
                                cW=cW.cpu().numpy(), cb=cb.cpu().numpy())
            weights_meta.write_text(json.dumps({"signature": signature, "layer": layer,
                "train_dir": str(train_dir), "n_train": 10000, "seeds": [0, 1, 2],
                "optimizer": "sgd", "lr": .001, "epochs": 10000}) + "\n")
            del X, Xk, probes, controls, hidden
            torch.cuda.empty_cache()
        for seed in (7, 11, 13):
            regime = "cot" if seed == 7 else f"cot_s{seed}"
            destination = root / regime / f"{role}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                saved = json.loads(destination.read_text())
                if saved.get("complete") and saved["weight_signature"] == signature:
                    print(f"skip {destination}", flush=True); continue
            base = OUT_DIR / "runs" / args.model / "L3" / regime
            neutral = {r["set_id"]: r for r in records(base / "neutral/behavior.jsonl")}
            inc = records(base / f"incongruent@{role}/behavior.jsonl")
            selected, extra = [], set()
            for row in inc:
                boundary = value_boundary(row["generation"], instances[row["id"]].names[role])
                if boundary and boundary["value"] == row["lure"]:
                    extra.add(row["set_id"])
                if row["set_id"] in sample or row["set_id"] in extra:
                    selected.append(row)
            matched = sample | extra
            selected += [neutral[key] for key in sorted(matched)]
            train_ids = {x["set_id"] for x in train["instances"]}
            if train_ids & matched:
                raise ValueError("generated-chain evaluation overlaps probe training")
            demos = make_demos(3, "word", seed=seed)
            output, pending = [], []
            for row in selected:
                x = instances[row["id"]]
                boundary = value_boundary(row["generation"], x.names[role])
                record = {"id": x.id, "set_id": x.set_id, "condition": x.condition,
                    "target": role, "primary_sample": x.set_id in sample, "lure_augmentation": x.set_id in extra,
                    "true_value": x.values[role], "lure": row["lure"], "answer_correct": row["correct"],
                    "name": x.names[role], "written_value": boundary["value"] if boundary else None,
                    "boundary_found": boundary is not None, "boundary_valid": False}
                output.append(record)
                if boundary is None:
                    continue
                prompt = build_prompt(x, demos, regime)
                enc = tok(prompt + row["generation"], return_offsets_mapping=True, add_special_tokens=True)
                marker = len(prompt) + boundary["marker"]
                indices = [i for i, (a, b_) in enumerate(enc["offset_mapping"]) if a <= marker < b_]
                if not indices:
                    continue
                index = indices[-1]
                try:
                    assert_before_value(enc["offset_mapping"], index, len(prompt) + boundary["value_start"])
                except ValueError:
                    continue
                record["boundary_valid"] = True
                pending.append((record, enc["input_ids"][:index + 1]))
            for offset in range(0, len(pending), 8):
                chunk = pending[offset:offset + 8]
                T = max(len(ids) for _, ids in chunk)
                ids = torch.full((len(chunk), T), tok.pad_token_id, device="cuda", dtype=torch.long)
                attention = torch.zeros_like(ids)
                for i, (_, sequence) in enumerate(chunk):
                    ids[i, :len(sequence)] = torch.tensor(sequence, device="cuda")
                    attention[i, :len(sequence)] = 1
                with torch.no_grad():
                    hs = model(input_ids=ids, attention_mask=attention, output_hidden_states=True).hidden_states
                    X = torch.stack([hs[layer][i, len(sequence) - 1].float() for i, (_, sequence) in enumerate(chunk)])
                    lp = torch.log_softmax(torch.einsum("nd,kcd->knc", X, W) + b[:, None], dim=-1).cpu().numpy()
                    cp = (torch.einsum("nd,kcd->knc", X, cW) + cb[:, None]).argmax(-1).cpu().numpy()
                for i, (record, _) in enumerate(chunk):
                    predictions = lp[:, i].argmax(-1)
                    record["predictions"] = predictions.tolist()
                    record["probe_accuracy"] = float(np.mean(predictions == record["true_value"]))
                    control = hashlib.sha256(f"0:{record['name']}".encode()).digest()[0] % 10
                    record["control_accuracy"] = float(np.mean(cp[:, i] == control))
                    record["p_true"] = float(np.exp(lp[:, i, record["true_value"]]).mean())
                    if record["lure"] is not None:
                        record["margin"] = float(np.mean(lp[:, i, record["true_value"]] - lp[:, i, record["lure"]]))
                        if not np.isfinite(record["margin"]):
                            raise ValueError("non-finite generated-chain probe margin")
                del hs, X, lp, cp
            meta = {"model": args.model, "regime": regime, "role": role, "layer": layer,
                "gold_neutral_accuracy": float(np.mean([r["eval"]["neutral"]["accuracy"] for r in candidates[layer]])),
                "weight_signature": signature, "n_train": 10000, "sample_sets": sorted(sample),
                "augmented_lure_sets": sorted(extra), "records": output, "complete": True,
                "pre_value_boundary_checked": True, "probe_seeds": [0, 1, 2]}
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(meta, indent=2) + "\n"); temporary.replace(destination)
            print(f"completed {args.model} {regime} {role}: {len(output)} rows, {len(pending)} boundaries", flush=True)
        del W, b, cW, cb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
