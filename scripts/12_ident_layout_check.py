#!/usr/bin/env python
"""Identifier-scheme (E38) layout check with real tokenizers: demos from the dataset's own pool,
two-token names allowed when consistent within a group. Runs on a dev node (tokenizers only)."""
import json, sys
sys.path.insert(0, "/work/jvl210002/migration/probing/src")
from pathlib import Path
from transformers import AutoTokenizer
from cueconf.config import DATA_DIR, model_entry
from cueconf.generator import read_jsonl
from cueconf.prompts import make_demos, scheme_of, layout, tokenize_layout, demo_seed_of, parse_regime
d = DATA_DIR / "L3_ident"; man = json.loads((d / "manifest.json").read_text())
for mk in ("llama32-3b", "olmo2-7b-it", "gemma3-4b-it"):
    tok = AutoTokenizer.from_pretrained(model_entry(mk)["hf_id"])
    xs = [x for x in read_jsonl(d / "test_sets.jsonl")][:400]
    for regime in ("cot_s7", "cot_s11", "direct_s13"):
        for cond in ("neutral", "incongruent@v1", "letter"):
            inst = [x for x in xs if (x.condition + (f"@{x.target}" if x.target else "")) == cond][:40]
            demos = make_demos(3, scheme_of(inst[0].condition), n=parse_regime(regime)[1], seed=demo_seed_of(regime), pool=man["demo_words"])
            toks = [tokenize_layout(tok, layout(x, demos, regime)) for x in inst]
            ref = toks[0]
            bad = sum(1 for t in toks if t["n_tokens"] != ref["n_tokens"] or t["name_ntok"] != ref["name_ntok"])
            print(f"{mk:14s} {regime:10s} {cond:16s} n_tokens={ref['n_tokens']} name_ntok={ref['name_ntok']} demo_names={[v for dd in demos for v in dd.names.values()][:4]} inconsistent={bad}/{len(toks)}")
