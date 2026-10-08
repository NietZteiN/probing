#!/usr/bin/env python
"""CPU validation gate for the reviewer GPU follow-ups."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "codecue/src")]
from transformers import AutoTokenizer
from codecue.config import model_entry as cm
from codecue.followups import control_demo, control_prompt
from codecue.generator import read_jsonl as cr
from codecue.prompts import build_prompt as cb, demo_block, demos
from cueconf.config import model_entry as am
from cueconf.followups import value_boundary
from cueconf.generator import read_jsonl as ar
from cueconf.prompts import assert_before_value, build_prompt, make_demos


def main():
    report = {"code": [], "arithmetic": []}
    example = next(x for x in cr(ROOT.parent / "codecue/data/L5/test_sets.jsonl") if x.condition == "neutral")
    for model in ("olmo2-7b-it", "llama32-3b-it"):
        tok = AutoTokenizer.from_pretrained(cm(model)["hf_id"], local_files_only=True)
        for seed in (7, 11, 13):
            ds = demos(5, seed, "trace")
            for kind in ("neutral_annotation", "numeric_elaboration"):
                p = control_prompt(example, ds, kind, tok)
                n = len(tok(p, add_special_tokens=True)["input_ids"])
                source = cb(example, "trace_expr", ds)
                ns = len(tok(source, add_special_tokens=True)["input_ids"])
                if kind == "neutral_annotation" and n != ns:
                    raise ValueError(f"full prompt length mismatch: {model} {seed}: {n} != {ns}")
                report["code"].append({"model": model, "seed": seed, "kind": kind,
                    "tokens": n, "source_expression_tokens": ns})
    for model in ("olmo2-1b-it", "llama32-3b"):
        tok = AutoTokenizer.from_pretrained(am(model)["hf_id"], local_files_only=True)
        for seed in (7, 11, 13):
            regime = "cot" if seed == 7 else f"cot_s{seed}"
            demos_ = make_demos(3, "word", seed=seed)
            for role in ("v1", "v2"):
                path = Path("/scratch/juno/jvl210002/probing/runs") / model / "L3" / regime / f"incongruent@{role}/behavior.jsonl"
                xs = {x.id:x for x in ar(ROOT / "data/L3/test_sets.jsonl")}
                checked, excluded, missing = 0, [], 0
                for line in list(path.open())[:32]:
                    row = json.loads(line); x = xs[row["id"]]
                    boundary = value_boundary(row["generation"], x.names[role])
                    if boundary is None:
                        missing += 1
                        continue
                    p = build_prompt(x, demos_, regime)
                    enc = tok(p + row["generation"], return_offsets_mapping=True, add_special_tokens=True)
                    marker = len(p) + boundary["marker"]
                    index = next(i for i,(a,b) in enumerate(enc["offset_mapping"]) if a <= marker < b)
                    try:
                        assert_before_value(enc["offset_mapping"], index, len(p) + boundary["value_start"])
                    except ValueError as error:
                        excluded.append({"id":row["id"], "reason":str(error),
                            "boundary":boundary, "token_offsets":enc["offset_mapping"][index],
                            "token_text":tok.decode([enc["input_ids"][index]])})
                        continue
                    checked += 1
                if not checked:
                    raise ValueError(f"no safe generated boundary for {model} {regime} {role}")
                report["arithmetic"].append({"model":model,"seed":seed,"role":role,
                    "checked":checked,"missing":missing,"excluded":excluded})
    report["status"] = "passed"
    log = ROOT / "log/round5_2026-10-03"
    (log / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    (log / "gpu_preflight.ok").write_text("Tokenizer, controls and value boundaries passed.\n")
    print("Round-five GPU preflight passed", flush=True)


if __name__ == "__main__":
    main()
