"""GPU stage 3: activation patching at the misleading name's token positions (plan §4.9).

Protocol (Kudo et al. §4.1, Zhang & Nanda 2024): force-decode prompt + gold chain for the
DESTINATION instance, replace the residual-stream activations at the patched layer(s) and
token positions with those cached from the SOURCE twin, and read the greedy next token at each
target position (`cotpre@target` = the step where the chain writes the target's value, and
`anspre` = the final answer). Twins have identical token layouts (rule 4 + fixed demos), so a
position in the source is the same position in the destination; this is asserted.

Contrasts (all pre-registered in PREREGISTRATION.md):
    main       source neutral         -> dest incongruent    recovery of lure errors
    ctl_word   source neutral_alt     -> dest neutral        patching damage (should stay correct)
    ctl_lure   source incongruent_alt -> dest incongruent    does the answer follow the NEW lure?

Sweep: every single layer, plus windows of 4 (Kudo's grid) and "all layers" as the ceiling.
Patch positions: every token carrying the target's name in the forced text (definition, uses,
chain restatements), or the prompt-only subset (`--scope prompt`).

Metrics per (layer set): recovery = P(correct after | lure error before); flip-to-new-lure for
ctl_lure; damage = P(wrong after | correct before) for ctl_word.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .generator import Instance
from .prompts import demo_seed_of, layout, make_demos, parse_regime, scheme_of, tokenize_layout
from .patch_summary import summarize, summarize_grid  # noqa: F401  (torch-free; importable on the login node)
from .runner import digit_token_ids


def decoder_layers(model):
    for attr in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "model.text_model.layers", "transformer.h"):
        obj = model
        try:
            for a in attr.split("."):
                obj = getattr(obj, a)
            return list(obj)
        except AttributeError:
            continue
    raise AttributeError("cannot find decoder layers on this model")


@contextmanager
def patch_hooks(model, layer_ids: Sequence[int], positions: Sequence[int], source: dict[int, torch.Tensor]):
    """source[layer] : [n_pos, d] activations to write at `positions` in that layer's output."""
    layers = decoder_layers(model)
    handles = []
    pos_t = torch.tensor(list(positions))

    def make_hook(l):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs[:, pos_t, :] = source[l].to(hs.dtype).to(hs.device)
            return output
        return hook

    for l in layer_ids:
        handles.append(layers[l].register_forward_hook(make_hook(l)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def hidden_at(model, input_ids: list[int], layer_ids: Sequence[int], positions: Sequence[int]) -> dict[int, torch.Tensor]:
    """Residual stream AFTER decoder layer l (== HF hidden_states[l+1]) at positions."""
    out = model(input_ids=torch.tensor([input_ids], device=model.device), output_hidden_states=True)
    return {l: out.hidden_states[l + 1][0, list(positions), :].clone() for l in layer_ids}


@torch.no_grad()
def next_digit(model, tok_digits: dict[str, list[int]], input_ids: list[int], read_positions: Sequence[int],
               return_scores: bool = False):
    """Greedy digit at each read position; with return_scores also the per-digit log-probs, from
    which the Zhang & Nanda (2024) logit difference logit(true) - logit(lure) is computed (E24)."""
    out = model(input_ids=torch.tensor([input_ids], device=model.device))
    res, allscores = [], []
    for p in read_positions:
        lp = torch.log_softmax(out.logits[0, p].float(), -1)
        scores = {d: float(torch.logsumexp(lp[ids], 0)) if ids else -1e9 for d, ids in tok_digits.items()}
        res.append(max(scores, key=scores.get)); allscores.append(scores)
    return (res, allscores) if return_scores else res


def layer_sets(n_layers: int, window: int = 4) -> list[tuple[str, list[int]]]:
    sets = [(f"L{l}", [l]) for l in range(n_layers)]
    sets += [(f"W{a}-{min(a + window, n_layers) - 1}", list(range(a, min(a + window, n_layers)))) for a in range(0, n_layers, window)]
    sets.append(("ALL", list(range(n_layers))))
    return sets


def run_contrast(tok, model, level: int, regime: str, pairs: list[tuple[Instance, Instance]], name: str,
                 out_path: Path, scope: str = "all", window: int = 4, only_ids: set[str] | None = None) -> dict:
    """pairs: (source, dest) twins. Writes per-pair, per-layer-set predictions."""
    n_layers = len(decoder_layers(model))
    sets = layer_sets(n_layers, window)
    dtoks = digit_token_ids(tok)
    demos = make_demos(level, scheme_of(pairs[0][1].condition), n=parse_regime(regime)[1], seed=demo_seed_of(regime))
    rows = []
    for src, dst in pairs:
        if only_ids is not None and dst.id not in only_ids:
            continue
        ls, ld = layout(src, demos, regime), layout(dst, demos, regime)
        ts, td = tokenize_layout(tok, ls), tokenize_layout(tok, ld)
        assert ts["n_tokens"] == td["n_tokens"] and ts["positions"] == td["positions"], (src.id, dst.id)
        # the renamed slot: the destination's target, or the source's when the destination is the
        # neutral twin (ctl_word: neutral_alt -> neutral)
        target = dst.target or src.target
        name_pos = td["name_tokens"][target]
        if scope == "prompt":
            name_pos = [p for p in name_pos if p < td["n_prompt_tokens"]]
        assert ts["name_tokens"][target] == td["name_tokens"][target]
        read_labels = [l for l in (f"cotpre@{target}", "anspre") if td["positions"].get(l) is not None]
        read_pos = [td["positions"][l] for l in read_labels]
        read_pos_dedup = sorted(set(read_pos))
        # what the chain SHOULD say at each read position
        gold = {f"cotpre@{target}": str(dst.values[target]), "anspre": str(dst.answer)}
        src_hidden = hidden_at(model, ts["input_ids"], range(n_layers), name_pos)
        b_dig, b_sc = next_digit(model, dtoks, td["input_ids"], read_pos, return_scores=True)
        base = dict(zip(read_labels, b_dig))
        # logit difference true - lure at each read position (lure defined only for incongruent destinations)
        def ld(scores):
            return {lab: (sc[gold[lab]] - sc[str(dst.lure)]) if dst.lure is not None else None for lab, sc in zip(read_labels, scores)}
        # the SOURCE run's own LD serves as the "clean" reference for normalisation
        s_dig, s_sc = next_digit(model, dtoks, ts["input_ids"], read_pos, return_scores=True)
        rec = {"src": src.id, "dst": dst.id, "set_id": dst.set_id, "target": target,
               "lure": dst.lure if dst.lure is not None else src.lure,   # inject: the lure comes from the source
               "src_lure": src.lure, "n_patched_tokens": len(name_pos), "gold": {k: gold[k] for k in read_labels},
               "base": base, "src_pred": dict(zip(read_labels, s_dig)), "base_ld": ld(b_sc), "src_ld": ld(s_sc),
               "patched": {}, "patched_ld": {}}
        for sname, lids in sets:
            with patch_hooks(model, lids, name_pos, {l: src_hidden[l] for l in lids}):
                p_dig, p_sc = next_digit(model, dtoks, td["input_ids"], read_pos, return_scores=True)
                rec["patched"][sname] = dict(zip(read_labels, p_dig))
                rec["patched_ld"][sname] = ld(p_sc)
        rows.append(rec)
        print(f"{name} {dst.id} base={base} patched[ALL]={rec['patched']['ALL']}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, sets)
    out_path.write_text(json.dumps({"contrast": name, "level": level, "regime": regime, "scope": scope,
                                    "layer_sets": [s for s, _ in sets], "summary": summary, "rows": rows}))
    return summary


# --------------------------------------------------------------------------- E29: Kudo grid
def run_grid_contrast(tok, model, level: int, regime: str, pairs: list[tuple[Instance, Instance]], name: str,
                      out_path: Path, window: int = 4, limit: int | None = None) -> dict:
    """Kudo et al. Figs. 5-6 in our setting: patch each equation / chain-step SPAN (all its tokens)
    at each `window`-layer block from the source into the destination, and read the greedy digit
    at the destination's target value step and final answer. Sources: a different neutral
    problem (their design), the neutral twin, or the alternative-lure twin; the caller builds the
    pairs. Metrics per cell: for twins, lure removal / follows-new-lure / normalised LD; for a
    different problem, Kudo's success rate = the answer becomes the SOURCE's answer."""
    n_layers = len(decoder_layers(model))
    windows = [(f"W{a}-{min(a + window, n_layers) - 1}", list(range(a, min(a + window, n_layers)))) for a in range(0, n_layers, window)]
    dtoks = digit_token_ids(tok)
    demos = make_demos(level, scheme_of(pairs[0][1].condition), n=parse_regime(regime)[1], seed=demo_seed_of(regime))
    rows = []
    for src, dst in (pairs[:limit] if limit else pairs):
        ls, ld_ = layout(src, demos, regime), layout(dst, demos, regime)
        ts, td = tokenize_layout(tok, ls), tokenize_layout(tok, ld_)
        assert ts["n_tokens"] == td["n_tokens"] and ts["segment_tokens"].keys() == td["segment_tokens"].keys(), (src.id, dst.id)
        target = dst.target or src.target
        read_labels = [l for l in (f"cotpre@{target}", "anspre") if target and td["positions"].get(l) is not None] or ["anspre"]
        read_pos = [td["positions"][l] for l in read_labels]
        gold = {f"cotpre@{target}": str(dst.values[target]) if target else None, "anspre": str(dst.answer)}
        src_gold = {f"cotpre@{target}": str(src.values[target]) if target else None, "anspre": str(src.answer)}
        b_dig, b_sc = next_digit(model, dtoks, td["input_ids"], read_pos, return_scores=True)
        def ld(scores):
            return {lab: (sc[gold[lab]] - sc[str(dst.lure)]) if dst.lure is not None else None for lab, sc in zip(read_labels, scores)}
        rec = {"src": src.id, "dst": dst.id, "set_id": dst.set_id, "target": target, "lure": dst.lure, "src_lure": src.lure,
               "gold": {k: gold[k] for k in read_labels}, "src_gold": {k: src_gold[k] for k in read_labels},
               "base": dict(zip(read_labels, b_dig)), "base_ld": ld(b_sc), "cells": {}}
        segs = td["segment_tokens"]
        src_hidden_all = hidden_at(model, ts["input_ids"], range(n_layers), sorted({t for v in segs.values() for t in v}))
        all_pos = sorted({t for v in segs.values() for t in v})
        idx = {t: i for i, t in enumerate(all_pos)}
        for seg_lab, seg_toks in segs.items():
            sel = [idx[t] for t in seg_toks]
            for wname, lids in windows:
                with patch_hooks(model, lids, seg_toks, {l: src_hidden_all[l][sel] for l in lids}):
                    p_dig, p_sc = next_digit(model, dtoks, td["input_ids"], read_pos, return_scores=True)
                rec["cells"][f"{seg_lab}|{wname}"] = {"pred": dict(zip(read_labels, p_dig)), "ld": ld(p_sc)}
        rows.append(rec)
        print(f"{name} {dst.id} base={rec['base']}", flush=True)
    summary = summarize_grid(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"contrast": name, "level": level, "regime": regime, "window": window,
                                    "summary": summary, "rows": rows}))
    return summary


