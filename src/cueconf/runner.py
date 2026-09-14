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
from .prompts import (demo_seed_of, layout, make_demos, parse_answer, parse_answer_free, parse_regime,
                      scheme_of, tokenize_layout)

DIGIT_STRS = [str(d) for d in range(10)]


def load_model(hf_id: str, dtype=torch.bfloat16, device: str = "cuda"):
    """Loads any causal LM in the panel. Gemma-3 checkpoints are multimodal
    (Gemma3ForConditionalGeneration): AutoModelForCausalLM returns the wrapper, whose
    `config.text_config` holds the layer count and width and whose `.model.language_model` is the
    text tower. We keep the wrapper (its forward accepts input_ids alone and returns logits) and
    read the text config where shapes are needed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, device_map=device)
    except ValueError:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(hf_id, dtype=dtype, device_map=device)
    model.eval()
    return tok, model


def text_config(model):
    """Shape-carrying config: the text tower's for multimodal checkpoints, the model's otherwise."""
    return getattr(model.config, "text_config", model.config)


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
def generate_free(tok, model, prompts: Sequence[str], max_new_tokens: int, batch_size: int,
                  stop_at_newline: bool = True) -> list[str]:
    outs: list[str] = []
    # stop at the end of the output line: "\n", "\n\n" and the newline token after a letter/digit
    nl_ids = sorted({i for s in ("\n", "\n\n", "a\n", "6\n", "6\n\n") for i in tok(s, add_special_tokens=False)["input_ids"]
                     if "\n" in tok.decode([i])})
    if stop_at_newline is False:
        nl_ids = []
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
                  layers: Sequence[int] | None = None, do_free: bool = True, do_forced: bool = True,
                  all_positions: bool = False) -> dict:
    """all_positions: cache EVERY token of the instance region (Kudo et al. style, E27) instead of
    the labelled positions; position labels become t<i> with i counted from the instance start,
    and meta records the token strings of the first instance for axis labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    demos = make_demos(level, scheme_of(condition), n=parse_regime(regime)[1], seed=demo_seed_of(regime))
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

    t0i = toks[keep_idx[0]]["instance_start_token"]
    summary = {"model": model_key, "level": level, "regime": regime, "condition": condition,
               "n": len(keep_idx), "excluded": excluded, "labels": labels,
               "positions": {k: toks[keep_idx[0]]["positions"][k] for k in labels},
               "n_tokens": ref[0], "n_prompt_tokens": toks[keep_idx[0]]["n_prompt_tokens"],
               "instance_start_token": t0i, "token_strings": toks[keep_idx[0]]["token_strings"],
               "segment_tokens": toks[keep_idx[0]]["segment_tokens"], "all_positions": all_positions}

    # ---- free generation (behaviour)
    if do_free:
        prompts = [lays[k].prompt for k in keep_idx]
        gens = generate_free(tok, model, prompts, max_new_tokens, batch_size,
                             stop_at_newline=parse_regime(regime)[0] != "free")
        with (out_dir / "behavior.jsonl").open("w") as f:
            n_correct = n_lure = 0
            for k, g in zip(keep_idx, gens):
                x = instances[k]
                pred = (parse_answer_free if parse_regime(regime)[0] == "free" else parse_answer)(g, x.names[x.query])
                correct = pred == x.answer
                is_lure = (x.lure is not None and pred == x.lure)
                n_correct += correct; n_lure += is_lure
                f.write(json.dumps({"id": x.id, "set_id": x.set_id, "condition": condition, "target": x.target,
                                    "lure": x.lure, "answer": x.answer, "pred": pred, "correct": correct,
                                    "pred_is_lure": is_lure, "generation": g, "values": x.values, "names": x.names,
                                    "query": x.query, "level": x.level}) + "\n")
        summary["free_accuracy"] = n_correct / max(1, len(keep_idx))
        summary["free_lure_rate"] = n_lure / max(1, len(keep_idx))

    # ---- forced pass (hidden states + next-token log-probs at *pre positions)
    if do_forced:
        pre_labels = [l for l in labels if (l.startswith("cotpre@") or l == "anspre" or l == "query") and summary["positions"][l] is not None]
        if all_positions:
            pos_labels = [f"t{i}" for i in range(ref[0] - t0i)]
            pos_list = [list(range(t0i, ref[0])) for _ in keep_idx]
        else:
            pos_labels = [l for l in labels if summary["positions"][l] is not None]
            pos_list = [[toks[k]["positions"][l] for l in pos_labels] for k in keep_idx]
        pre_list = [[toks[k]["positions"][l] for l in pre_labels] for k in keep_idx]
        ids_list = [toks[k]["input_ids"] for k in keep_idx]
        dtoks = digit_token_ids(tok)
        n_layers_total = text_config(model).num_hidden_layers + 1
        keep_layers = list(range(n_layers_total)) if layers is None else list(layers)
        d = text_config(model).hidden_size
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
