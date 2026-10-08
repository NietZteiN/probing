#!/usr/bin/env python
"""CPU preflight: supplied target digits must be excluded from pre-value tokens.

Every cached row is also guarded during the GPU pass. This samples the actual train/test
corpora for all five tokenizers and all three code demonstration sets before those jobs run.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parent / "codecue"
sys.path[:0] = [str(ROOT / "src"), str(CODE / "src")]

from transformers import AutoTokenizer
from cueconf.config import DATA_DIR as ADATA, model_entry as amodel
from cueconf.generator import read_jsonl as aread
from cueconf.prompts import layout, make_demos, tokenize_layout
from codecue.cache import cache_spans, trace_text
from codecue.config import DATA_DIR as CDATA, model_entry as cmodel
from codecue.generator import read_jsonl as cread
from codecue.layout import assert_before_value, spans_to_tokens
from codecue.probe_data import disjoint_training
from codecue.prompts import build_prompt, demos


def main():
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "checks": []}
    code_test = cread(CDATA / "L5/test_sets.jsonl")
    code_train = disjoint_training(cread(CDATA / "L5/train_neutral.jsonl"), code_test, 2000)
    groups = [code_train[:32]]
    for condition in ("neutral", "congruent", "incongruent", "incongruent_alt"):
        groups.append([x for x in code_test if x.condition == condition and x.target in (None, "v1")][:32])
    for model in ("olmo2-7b-it", "llama32-3b-it", "llama31-8b-it"):
        tok = AutoTokenizer.from_pretrained(cmodel(model)["hf_id"], local_files_only=True)
        for seed in (7, 11, 13):
            regime = "trace" if seed == 7 else f"trace_s{seed}"
            dem = demos(5, seed, "trace")
            n = 0
            for group in groups:
                for x in group:
                    prompt, gold = build_prompt(x, regime, dem), trace_text(x)
                    enc = tok(prompt + gold, return_offsets_mapping=True, add_special_tokens=True)
                    mapped = spans_to_tokens(cache_spans(prompt, gold, x, "v1"), enc["offset_mapping"])
                    step = re.search(rf"\b{re.escape(x.names['v1'])} = ", gold)
                    assert_before_value(enc["offset_mapping"], mapped["pre@v1"], len(prompt) + step.end())
                    assert_before_value(enc["offset_mapping"], mapped["anspre"], len(prompt + gold) - len(str(x.answer)))
                    n += 1
            report["checks"].append({"project": "codecue", "model": model, "regime": regime, "n": n, "passed": True})
            print(f"code {model} {regime}: {n} passed", flush=True)
    arith_train = aread(ADATA / "L3/probe_train_neutral.jsonl")[:32]
    arith_test = aread(ADATA / "L3/test_sets.jsonl")
    arith = arith_train + [x for x in arith_test if x.condition == "neutral"][:32]
    for role in ("v1", "v2"):
        arith += [x for x in arith_test if x.condition == "incongruent" and x.target == role][:32]
    for model in ("llama32-3b-it", "llama31-8b-it", "gemma3-4b-it", "gemma3-12b-it"):
        tok = AutoTokenizer.from_pretrained(amodel(model)["hf_id"], local_files_only=True)
        for regime in ("cot", "direct"):
            dem = make_demos(3, "word", seed=7)
            for x in arith:
                tokenize_layout(tok, layout(x, dem, regime))
            report["checks"].append({"project": "probing", "model": model, "regime": regime, "n": len(arith), "passed": True})
            print(f"arithmetic {model} {regime}: {len(arith)} passed", flush=True)
    report["status"] = "passed"
    path = ROOT / "log/round4_2026-10-02/value_boundaries.json"
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
