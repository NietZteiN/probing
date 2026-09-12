"""GPU stage 1: behaviour, hidden-state caching, forced-decoding logits.

For one (model, level, regime) and a list of instance files:

  1. FREE generation, greedy, stop at newline -> the model's own answer (behaviour: accuracy,
     lure rate). Saved per instance in `behavior.jsonl`.
  2. FORCED pass over prompt + gold output -> residual-stream hidden states at every layer at the
     labelled positions (prompts.layout), saved as float16 memmaps, one file per condition:
        hidden.npy   [n_inst, n_pos, n_layers+1, d]   (index 0 = embeddings, as in HF)
        meta.json    labels, position order, layer count, per-instance values/lure/target
     plus the next-token distribution at every `*pre` position (`forced_logits.jsonl`: top-10
     tokens with log-probs, and the log-prob of the true and lure digit tokens). The forced
     pass is what probes and the crossover analysis read; it is also the "teacher-forced
     behaviour" (does the model write the lure at P4/P5 when the chain so far is correct?).

Positions P1-P3 are in the prompt, so they are identical between the free and forced runs.

Layout guard: every instance of a (condition, regime) must tokenize to the same length with the
same position indices (rule 4 + fixed demos). Instances that violate it are EXCLUDED and
counted in meta["excluded"]; the job fails if more than 1% are excluded, because that means a
word list was not checked for this tokenizer.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .generator import Instance, read_jsonl
from .prompts import layout, make_demos, parse_answer, scheme_of, tokenize_layout

DIGIT_STRS = [str(d) for d in range(10)]


def load_model(hf_id: str, dtype=torch.bfloat16, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, device_map=device)
    model.eval()
    # gemma-3 multimodal checkpoints: the text tower is model.language_model
    return tok, model


def digit_token_ids(tok) -> dict[str, list[int]]:
    """Token ids that decode to a bare digit, with and without a leading space. In our layout the
    value follows '=' directly, so the no-space variant is the one that matters; both are kept."""
    out = {}
    for d in DIGIT_STRS:
        ids = set()
        for s in (d, " " + d):
            enc = tok(s, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:
                ids.add(enc[0])
        out[d] = sorted(ids)
    return out


@torch.no_grad()
def generate_free(tok, model, prompts: Sequence[str], max_new_tokens: int, batch_size: int) -> list[str]:
    outs: list[str] = []
    # stop at the end of the output line: "\n", "\n\n" and the newline token after a letter/digit
    nl_ids = sorted({i for s in ("\n", "\n\n", "a\n", "6\n", "6\n\n") for i in tok(s, add_special_tokens=False)["input_ids"]
                     if "\n" in tok.decode([i])})
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=nl_ids + [tok.eos_token_id])
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    return outs


@torch.no_grad()
def forced_pass(tok, model, input_ids_list: Sequence[list[int]], positions_list: Sequence[list[int]],
                pre_positions_list: Sequence[list[int]], batch_size: int, layers: Sequence[int] | None = None):
    """Yields (hidden [n_pos, n_layers_kept, d] float16 numpy, logprobs at pre positions [n_pre, V]
    float32 numpy) per instance, right-padded so positions are absolute."""
    pad = tok.pad_token_id
    for i in range(0, len(input_ids_list), batch_size):
        ids = input_ids_list[i:i + batch_size]
        L = max(len(x) for x in ids)
        inp = torch.full((len(ids), L), pad, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for b, x in enumerate(ids):
            inp[b, :len(x)] = torch.tensor(x); att[b, :len(x)] = 1
        out = model(input_ids=inp.to(model.device), attention_mask=att.to(model.device), output_hidden_states=True)
        hs = out.hidden_states                          # tuple (n_layers+1) of [B, L, d]
        keep = list(range(len(hs))) if layers is None else list(layers)
        stacked = torch.stack([hs[l] for l in keep], dim=2)   # [B, L, nL, d]
        logp = torch.log_softmax(out.logits.float(), dim=-1)  # [B, L, V]
        for b in range(len(ids)):
            pos = positions_list[i + b]
            pre = pre_positions_list[i + b]
            h = stacked[b, pos].to(torch.float16).cpu().numpy()
            lp = logp[b, pre].cpu().numpy() if pre else np.zeros((0, logp.shape[-1]), dtype=np.float32)
            yield h, lp


def run_condition(tok, model, model_key: str, level: int, regime: str, condition: str,
                  instances: list[Instance], out_dir: Path, batch_size: int, max_new_tokens: int,
                  layers: Sequence[int] | None = None, do_free: bool = True, do_forced: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    demos = make_demos(level, scheme_of(condition))
    t0 = time.time()
    lays = [layout(x, demos, regime) for x in instances]
    toks = [tokenize_layout(tok, lay) for lay in lays]

    # ---- layout guard
    labels = list(lays[0].positions)
    ref = (toks[0]["n_tokens"], tuple(toks[0]["positions"][k] for k in labels))
    keep_idx, excluded = [], []
    for k, t in enumerate(toks):
        sig = (t["n_tokens"], tuple(t["positions"][kk] for kk in labels))
        bad_names = [r for r, v in t["name_ntok"].items() if v != [1]]
        if sig != ref or bad_names:
            excluded.append({"id": instances[k].id, "n_tokens": t["n_tokens"], "multi_token_names": bad_names})
        else:
            keep_idx.append(k)
    if len(excluded) > 0.01 * len(instances):
        raise RuntimeError(f"{model_key} {condition} {regime}: {len(excluded)}/{len(instances)} instances break the "
                           f"token layout; run 11_tokenizer_check.py for this model. First: {excluded[:3]}")

    summary = {"model": model_key, "level": level, "regime": regime, "condition": condition,
               "n": len(keep_idx), "excluded": excluded, "labels": labels,
               "positions": {k: toks[keep_idx[0]]["positions"][k] for k in labels},
               "n_tokens": ref[0], "n_prompt_tokens": toks[keep_idx[0]]["n_prompt_tokens"]}

    # ---- free generation (behaviour)
    if do_free:
        prompts = [lays[k].prompt for k in keep_idx]
        gens = generate_free(tok, model, prompts, max_new_tokens, batch_size)
        with (out_dir / "behavior.jsonl").open("w") as f:
            n_correct = n_lure = 0
            for k, g in zip(keep_idx, gens):
                x = instances[k]
                pred = parse_answer(g, x.names[x.query])
                correct = pred == x.answer
                is_lure = (x.lure is not None and pred == x.lure)
                n_correct += correct; n_lure += is_lure
                f.write(json.dumps({"id": x.id, "set_id": x.set_id, "condition": condition, "target": x.target,
                                    "lure": x.lure, "answer": x.answer, "pred": pred, "correct": correct,
                                    "pred_is_lure": is_lure, "generation": g, "values": x.values}) + "\n")
        summary["free_accuracy"] = n_correct / max(1, len(keep_idx))
        summary["free_lure_rate"] = n_lure / max(1, len(keep_idx))

    # ---- forced pass (hidden states + next-token log-probs at *pre positions)
    if do_forced:
        pre_labels = [l for l in labels if (l.startswith("cotpre@") or l == "anspre" or l == "query") and summary["positions"][l] is not None]
        pos_labels = [l for l in labels if summary["positions"][l] is not None]
        pos_list = [[toks[k]["positions"][l] for l in pos_labels] for k in keep_idx]
        pre_list = [[toks[k]["positions"][l] for l in pre_labels] for k in keep_idx]
        ids_list = [toks[k]["input_ids"] for k in keep_idx]
        dtoks = digit_token_ids(tok)
        n_layers_total = model.config.num_hidden_layers + 1 if not hasattr(model.config, "text_config") else model.config.text_config.num_hidden_layers + 1
        keep_layers = list(range(n_layers_total)) if layers is None else list(layers)
        d = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size
        hidden = np.lib.format.open_memmap(out_dir / "hidden.npy", mode="w+", dtype=np.float16,
                                           shape=(len(keep_idx), len(pos_labels), len(keep_layers), d))
        fl = (out_dir / "forced_logits.jsonl").open("w")
        for n, (h, lp) in enumerate(forced_pass(tok, model, ids_list, pos_list, pre_list, batch_size, keep_layers)):
            hidden[n] = h
            x = instances[keep_idx[n]]
            rec = {"id": x.id, "pre": {}}
            for j, lab in enumerate(pre_labels):
                row = lp[j]
                digit_lp = {dd: float(np.logaddexp.reduce(row[ids])) if ids else -math.inf for dd, ids in dtoks.items()}
                top = np.argsort(-row)[:10]
                rec["pre"][lab] = {"digit_logp": digit_lp, "top": [(int(t), float(row[t])) for t in top],
                                   "argmax_digit": max(digit_lp, key=digit_lp.get)}
            fl.write(json.dumps(rec) + "\n")
        fl.close()
        hidden.flush()
        summary.update({"pos_labels": pos_labels, "pre_labels": pre_labels, "layers": keep_layers, "hidden_dim": d,
                        "hidden_shape": list(hidden.shape)})
        (out_dir / "meta.json").write_text(json.dumps({**summary, "instances": [
            {"id": instances[k].id, "set_id": instances[k].set_id, "values": instances[k].values, "target": instances[k].target,
             "lure": instances[k].lure, "answer": instances[k].answer, "query": instances[k].query,
             "names": instances[k].names, "name_tokens": toks[k]["name_tokens"]} for k in keep_idx]}, indent=1))
    summary["seconds"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
