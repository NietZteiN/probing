#!/usr/bin/env python
"""Rule 4 per model: which candidate words are one token in every context, and how the prompt
template tokenizes. Needs a tokenizer -> run in a job (make tokcheck), never on the login node.

    python scripts/11_tokenizer_check.py --model llama32-3b [--model gemma3-4b ...]

Writes data/words/<model>.json and prints the token layout of one level-3 instance in both
regimes so a merged "=?" or ", " is visible before any hidden state is cached.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import DATA_DIR, load_config, model_entry  # noqa: E402
from cueconf.generator import sample_sets  # noqa: E402
from cueconf.prompts import layout, make_demos, tokenize_layout  # noqa: E402
from cueconf.tokcheck import check_words, template_report, write_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    models = load_config("models.yaml")["models"] if a.model == ["all"] else a.model
    rc = 0
    for key in models:
        m = model_entry(key)
        tok = AutoTokenizer.from_pretrained(m["hf_id"])
        rep = check_words(tok)
        rep["model"] = key; rep["hf_id"] = m["hf_id"]
        # template layout on one instance per regime
        s = next(sample_sets(3, 1, seed=1, pool=rep["neutral"] or ["pen", "cup", "box", "dog", "cat"]))
        inc = next(x for x in s if x.condition == "incongruent")
        rep["template"] = {}
        for regime in ("cot", "direct"):
            lay = layout(inc, make_demos(3, "word"), regime)
            tk = tokenize_layout(tok, lay)
            rep["template"][regime] = {"positions": tk["positions"], "name_ntok": tk["name_ntok"],
                                       "tokens_after_instance_start": [t for t in template_report(tok, lay.text)][-40:]}
            bad = [k for k, v in tk["positions"].items() if v is not None and lay.text[lay.positions[k]] not in tok.decode([tk["input_ids"][v]])]
            rep["template"][regime]["misaligned_labels"] = bad
            if bad:
                rc = 1
        write_report(DATA_DIR / "words" / f"{key}.json", rep)
        print(f"{key}: number words ok={rep['number_ok']} {rep['number_ntok']}")
        print(f"  neutral words passing rule 4: {len(rep['neutral'])}/{len(rep['neutral']) + len(rep['rejected'])}; rejected: {rep['rejected']}")
        for regime in ("cot", "direct"):
            t = rep["template"][regime]
            print(f"  [{regime}] positions={t['positions']} misaligned={t['misaligned_labels']}")
            print("  ", " | ".join(f"{i}:{s!r}" for i, s in t["tokens_after_instance_start"]))
        if not rep["number_ok"]:
            print(f"  WARNING: {key} splits a number word; the incongruent condition is not usable with this tokenizer")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
