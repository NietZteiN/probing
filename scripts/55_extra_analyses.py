#!/usr/bin/env python
"""Analysis-only experiments on the finished runs (login node):

  E19  lure-distance covariate: accuracy / lure rate in the incongruent condition as a function
       of |lure - true|, with a clustered logit `correct ~ distance`
  E20  teacher-forced lure rate: with the gold chain forced up to the target's value step, how
       often is the model's next digit the lure (from forced_logits.jsonl), vs the true value;
       compared with the free-generation lure rate
  E26  error analysis of the number-word-name effect in the chain regime: classify every wrong
       chain in neutral / congruent / incongruent by where it first departs from the gold chain
       (which step, and whether the written value equals the lure, an operand, another
       variable's value, or something else)

    python scripts/55_extra_analyses.py [--models ...] [--level 3]
Writes results/summary/<model>/L<level>/<regime>/extras.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, RESULTS_DIR, load_config  # noqa: E402
from cueconf.stats import bootstrap_ci, clustered_logit  # noqa: E402


def load_behavior(run_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for g in sorted(run_dir.iterdir()):
        f = g / "behavior.jsonl"
        if f.exists() and "__" not in g.name:
            out[g.name] = [json.loads(l) for l in f.open()]
    return out


def e19(beh: dict) -> dict:
    res = {}
    for g, rows in beh.items():
        if not g.startswith("incongruent@"):
            continue
        df = pd.DataFrame([{"set_id": r["set_id"], "correct": int(r["correct"]), "lure": int(r["pred_is_lure"]),
                            "dist": abs(r["lure"] - r["values"][r["target"]])} for r in rows])
        by = {int(d): {"n": int(len(s)), "acc": float(s["correct"].mean()), "lure_rate": float(s["lure"].mean())} for d, s in df.groupby("dist")}
        try:
            fit = clustered_logit(df, "correct ~ dist")
            coef = {"coef": float(fit.params["dist"]), "se": float(fit.bse["dist"]), "p": float(fit.pvalues["dist"])}
        except Exception as e:  # noqa: BLE001
            coef = {"error": str(e)}
        res[g] = {"by_distance": by, "logit_correct_on_distance": coef}
    return res


def e20(run_dir: Path, beh: dict) -> dict:
    res = {}
    for g, rows in beh.items():
        if not g.startswith("incongruent@"):
            continue
        fl = run_dir / g / "forced_logits.jsonl"
        if not fl.exists():
            continue
        forced = {json.loads(l)["id"]: json.loads(l)["pre"] for l in fl.open()}
        t = rows[0]["target"]
        lab = f"cotpre@{t}"
        n = n_lure = n_true = 0
        ans_n = ans_lure = 0
        for r in rows:
            pre = forced.get(r["id"], {})
            if lab in pre:
                n += 1
                d = pre[lab]["argmax_digit"]
                n_lure += d == str(r["lure"]); n_true += d == str(r["values"][t])
            if "anspre" in pre:
                ans_n += 1; ans_lure += pre["anspre"]["argmax_digit"] == str(r["lure"])
        res[g] = {"value_step": {"n": n, "forced_lure_rate": n_lure / n if n else None, "forced_true_rate": n_true / n if n else None},
                  "answer": {"n": ans_n, "forced_lure_rate": ans_lure / ans_n if ans_n else None},
                  "free_lure_rate": float(np.mean([r["pred_is_lure"] for r in rows]))}
    return res


def _split_chain(s: str) -> list[str]:
    return [p.strip() for p in s.split("\n", 1)[0].split(",")]


def e26(beh: dict, gold_by_id: dict) -> dict:
    """Where does a wrong chain first depart from the gold chain, and what did it write?"""
    res = {}
    for g, rows in beh.items():
        if g in ("letter",) or g.startswith("train_"):
            continue
        cats = Counter(); step_of_first_error = Counter(); n_wrong = 0
        for r in rows:
            if r["correct"]:
                continue
            n_wrong += 1
            gold = _split_chain(gold_by_id[r["id"]]); gen = _split_chain(r["generation"])
            k = next((i for i, (a, b) in enumerate(zip(gold, gen)) if a != b), None)
            if k is None:
                cats["truncated_or_extra" if len(gen) != len(gold) else "unparsed"] += 1; step_of_first_error["none"] += 1
                continue
            step_of_first_error[f"step{k}"] += 1
            g_step, w_step = gold[k], gen[k]
            gl, _, gr = g_step.partition("="); wl, _, wr = w_step.partition("=")
            if gl != wl:
                cats["wrong_lhs_name"] += 1; continue
            # what value / expression was written instead
            vals = set(str(v) for v in r["values"].values())
            # operands per equation of the INPUT: gold[0] is the queried variable's equation restated
            first_eq_digits = set(ch for ch in gold[0].split("=", 1)[1] if ch.isdigit())
            own_eq_digits = set(ch for ch in g_step.split("=", 1)[1] if ch.isdigit()) if k > 0 else set()
            w = wr.strip()
            if r.get("lure") is not None and w == str(r["lure"]):
                cats["wrote_lure"] += 1
            elif "+" in wr or "-" in wr:
                cats["wrong_expression"] += 1
            elif w in first_eq_digits and k >= 1 and gl != gold[0].partition("=")[0]:
                cats["first_equation_operand"] += 1          # e.g. truck=2 + ant, ant=7 - 6 -> writes ant=2
            elif w in vals:
                cats["other_variables_value"] += 1
            elif w in own_eq_digits:
                cats["own_operand_digit"] += 1
            else:
                cats["other_value"] += 1
        res[g] = {"n_wrong": n_wrong, "first_error_step": dict(step_of_first_error), "categories": dict(cats)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=None); ap.add_argument("--level", type=int, default=3)
    a = ap.parse_args()
    from cueconf.config import DATA_DIR
    from cueconf.generator import read_jsonl
    gold = {x.id: x.cot for x in read_jsonl(DATA_DIR / f"L{a.level}" / "test_sets.jsonl")}
    gold_direct = {x.id: x.direct for x in read_jsonl(DATA_DIR / f"L{a.level}" / "test_sets.jsonl")}
    for k in (a.models or list(load_config("models.yaml")["models"])):
        for regime in ("cot", "direct"):
            run_dir = OUT_DIR / "runs" / k / f"L{a.level}" / regime
            if not run_dir.exists():
                continue
            beh = load_behavior(run_dir)
            if not beh:
                continue
            out = {"E19_lure_distance": e19(beh), "E20_teacher_forced": e20(run_dir, beh)}
            if regime == "cot":
                out["E26_error_analysis"] = e26(beh, gold)
            dst = RESULTS_DIR / "summary" / k / f"L{a.level}" / regime / "extras.json"
            dst.write_text(json.dumps(out, indent=1, default=float))
            print(f"{k} {regime}: wrote {dst}")
            if regime == "cot":
                for g, v in out["E26_error_analysis"].items():
                    print(f"   E26 {g:20s} wrong={v['n_wrong']:4d} first-error step={v['first_error_step']} cats={v['categories']}")
            for g, v in out["E20_teacher_forced"].items():
                print(f"   E20 {g:20s} forced lure@value-step={v['value_step']['forced_lure_rate']} forced lure@answer={v['answer']['forced_lure_rate']} free lure={v['free_lure_rate']:.3f}")
            for g, v in out["E19_lure_distance"].items():
                print(f"   E19 {g:20s} acc by |lure-true|: " + " ".join(f"{d}:{s['acc']:.2f}/{s['lure_rate']:.2f}" for d, s in sorted(v['by_distance'].items())) + f" | logit {v['logit_correct_on_distance']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
