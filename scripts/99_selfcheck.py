#!/usr/bin/env python
"""Adversarial self-check over everything on disk. Each check is a way the pipeline could have
produced a plausible, wrong table; the point is to fail loudly rather than to reassure.

    python scripts/99_selfcheck.py [--level 3] [--models ...]

Exit code 1 if any check fails. Warnings do not fail the run but are printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import DATA_DIR, OUT_DIR, RESULTS_DIR, load_config  # noqa: E402
from cueconf.generator import instance_arithmetic, read_jsonl  # noqa: E402

FAIL, WARN = [], []


def check(cond: bool, msg: str, warn: bool = False) -> None:
    if cond:
        return
    (WARN if warn else FAIL).append(msg)


def npy_header(path: Path):
    """(shape, dtype) without mapping the file: the login node caps virtual memory at 8 GB and a
    cache is several GB, so `np.load(mmap_mode='r')` dies with ENOMEM here."""
    import numpy.lib.format as fmt
    with path.open("rb") as f:
        major, minor = fmt.read_magic(f)
        reader = {(1, 0): fmt.read_array_header_1_0, (2, 0): fmt.read_array_header_2_0}[(major, minor)]
        shape, fortran, dtype = reader(f)
        return shape, dtype, f.tell()


def npy_first_row(path: Path, shape, dtype, offset: int) -> np.ndarray:
    """Just the first instance, read directly, so we can look for NaNs without mapping 5 GB."""
    n = int(np.prod(shape[1:]))
    with path.open("rb") as f:
        f.seek(offset)
        return np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)


def check_dataset(level: int, suffix: str = "") -> None:
    d = DATA_DIR / f"L{level}{suffix}"
    if not (d / "test_sets.jsonl").exists():
        return
    rows = read_jsonl(d / "test_sets.jsonl")
    by_set = defaultdict(dict)
    for x in rows:
        by_set[x.set_id][(x.condition, x.target)] = x
    tag = f"data L{level}{suffix}"
    # 1. every condition of a set shares arithmetic, values and answer
    bad = 0
    for sid, d_ in by_set.items():
        vals = {json.dumps(x.eqs, sort_keys=True) for x in d_.values()}
        ans = {x.answer for x in d_.values()}
        if len(vals) != 1 or len(ans) != 1:
            bad += 1
    check(bad == 0, f"{tag}: {bad} matched sets whose members differ in arithmetic or answer")
    # 2. a lure is never a value or an operand digit, and never equals the true answer
    bad = [x.id for x in rows if x.lure is not None and
           (x.lure in x.values.values() or x.lure in {int(a) for e in x.eqs for a in e["args"] if a.isdigit()})]
    check(not bad, f"{tag}: {len(bad)} instances whose lure is a value or an operand ({bad[:3]})")
    # 3. twins differ from the neutral twin in exactly one name
    bad = 0
    for sid, d_ in by_set.items():
        neu = d_.get(("neutral", None))
        if neu is None:
            continue
        for (cond, tgt), x in d_.items():
            if cond in ("neutral", "letter"):
                continue
            diff = [r for r in neu.names if neu.names[r] != x.names[r]]
            if diff != [tgt]:
                bad += 1
    check(bad == 0, f"{tag}: {bad} twins differ from neutral in other than exactly the target name")
    # 4. probe-train and test expressions are disjoint
    tr = read_jsonl(d / "probe_train_neutral.jsonl")
    tr_ex = set().union(*(instance_arithmetic(x).expressions() for x in tr))
    te_ex = set().union(*(instance_arithmetic(x).expressions() for x in rows))
    check(not (tr_ex & te_ex), f"{tag}: {len(tr_ex & te_ex)} digit expressions shared by probe-train and test")
    # 5. the answer really is what the chain writes last
    bad = [x.id for x in rows[:2000] if not x.cot.rstrip().endswith(f"{x.names[x.query]}={x.answer}")]
    check(not bad, f"{tag}: {len(bad)} gold chains whose last step is not the answer ({bad[:2]})")


def check_runs(model: str, level: int, suffix: str = "") -> None:
    base = OUT_DIR / "runs" / model / f"L{level}{suffix}"
    if not base.is_dir():
        return
    for regime_dir in sorted(base.iterdir()):
        regime = regime_dir.name
        tag = f"{model} L{level}{suffix} {regime}"
        groups = [g for g in sorted(regime_dir.iterdir()) if (g / "summary.json").exists()]
        pos_sig, layouts = {}, {}
        for g in groups:
            s = json.loads((g / "summary.json").read_text())
            name = g.name
            # 6. nothing was silently dropped
            check(len(s["excluded"]) == 0,
                  f"{tag} {name}: {len(s.get('excluded', []))} instances excluded by the layout guard", warn=True)
            check(s["n"] >= 1, f"{tag} {name}: empty group")
            # 7. every group of a regime has the SAME token layout, or probes compare apples to oranges
            layouts[name] = (s["n_tokens"], tuple(sorted((k, v) for k, v in s["positions"].items())))
            # 8. free-generation answers were parsed
            bf = g / "behavior.jsonl"
            if bf.exists():
                rows = [json.loads(l) for l in bf.open()]
                none = sum(1 for r in rows if r["pred"] is None)
                check(none / max(1, len(rows)) < 0.05,
                      f"{tag} {name}: {100*none/max(1,len(rows)):.1f}% of generations could not be parsed", warn=True)
                # 9. a lure flag must mean pred == lure
                bad = [r["id"] for r in rows if r["pred_is_lure"] != (r["lure"] is not None and r["pred"] == r["lure"])]
                check(not bad, f"{tag} {name}: {len(bad)} rows where pred_is_lure disagrees with pred==lure")
                # 10. accuracy in summary.json matches the rows
                acc = np.mean([r["correct"] for r in rows])
                check(abs(acc - s.get("free_accuracy", acc)) < 1e-6,
                      f"{tag} {name}: summary accuracy {s.get('free_accuracy')} != rows {acc:.4f}")
        # letter and word conditions use different demonstrations, so their prompts differ in
        # length by design; only groups within one naming scheme have to align token for token
        word_layouts = {k: v for k, v in layouts.items()
                        if not k.startswith("train_") and "__" not in k and k != "letter"}
        if len(set(word_layouts.values())) > 1:
            byl = defaultdict(list)
            for k, v in word_layouts.items():
                byl[v].append(k)
            FAIL.append(f"{tag}: word-named test groups have {len(byl)} token layouts; twins are not aligned: "
                        + "; ".join(f"{len(v)} groups at {k[0]} tokens" for k, v in byl.items()))
        # 11. hidden-state caches must match their meta
        for g in groups:
            h, m = g / "hidden.npy", g / "meta.json"
            if h.exists() and m.exists():
                meta = json.loads(m.read_text())
                shape, dtype, off = npy_header(h)
                check(list(shape) == meta["hidden_shape"],
                      f"{tag} {g.name}: hidden.npy shape {shape} != meta {meta['hidden_shape']}")
                check(len(meta["instances"]) == shape[0],
                      f"{tag} {g.name}: {len(meta['instances'])} meta instances for {shape[0]} cached rows")
                row = npy_first_row(h, shape, dtype, off).astype(np.float32)
                check(not np.isnan(row).any(), f"{tag} {g.name}: NaNs in the first cached instance")
                check(float(np.abs(row).max()) > 0, f"{tag} {g.name}: first cached instance is all zeros")
                check(shape[2] == len(meta["layers"]) and shape[1] == len(meta["pos_labels"]),
                      f"{tag} {g.name}: cache axes {shape} disagree with "
                      f"{len(meta['pos_labels'])} positions x {len(meta['layers'])} layers")


def check_probes(model: str, level: int) -> None:  # noqa: C901
    base = OUT_DIR / "probes" / model / f"L{level}"
    if not base.is_dir():
        return
    for regime_dir in sorted(base.iterdir()):
        for train_dir in sorted(regime_dir.iterdir()):
            for pj in sorted(train_dir.glob("*.json")):
                d = json.loads(pj.read_text())
                tag = f"{model} L{level} {regime_dir.name} {train_dir.name} {pj.stem}"
                res = d["results"]
                check(bool(res), f"{tag}: no probe results")
                if not res:
                    continue
                # 12. the probe must beat chance on the condition it was trained for
                best = max((r["eval"].get("neutral", {}).get("accuracy", 0) for r in res), default=0)
                distractor = load_config("dataset.yaml") and pj.stem == "v3" and level == 4
                check(best > 0.5 or distractor,
                      f"{tag}: best neutral accuracy only {best:.2f}; probes may not have trained", warn=True)
                # 13. seeds must actually differ
                by_cell = defaultdict(list)
                for r in res:
                    by_cell[(r["position"], r["layer"])].append(r["train_acc"])
                ident = sum(1 for v in by_cell.values() if len(v) > 1 and len(set(v)) == 1)
                check(ident < 0.9 * len(by_cell),
                      f"{tag}: {ident}/{len(by_cell)} cells identical across seeds; seeds may be ignored", warn=True)
                # 14. a lure rate exists exactly when the lure sits on THIS probe's variable
                role = d["role"]
                for r in res[:40]:
                    for cond, ev in r["eval"].items():
                        has_lure = ("@" in cond and cond.split("@")[1] == role
                                    and cond.split("@")[0] in ("incongruent", "incongruent_alt", "irrelevant"))
                        if has_lure and ev.get("lure_rate") is None:
                            FAIL.append(f"{tag}: {cond} targets {role} but has no lure rate")
                        if not has_lure and ev.get("lure_rate") is not None:
                            FAIL.append(f"{tag}: {cond} does not target {role} but has a lure rate")


def check_sweeps(level: int) -> None:
    p = RESULTS_DIR / "summary" / f"seed_sweep_L{level}.json"
    if not p.exists():
        return
    sw = json.loads(p.read_text())
    for key, rec in sw.items():
        tag = f"sweep L{level} {key}"
        # 15. a claimable contrast must have the same sign in every seed
        for name, c in rec["contrasts"].items():
            signs = {np.sign(v) for v in c["by_seed"].values()}
            if c["claimable"]:
                check(len(signs) == 1 and 0 not in signs and (c["ci95"][0] > 0 or c["ci95"][1] < 0),
                      f"{tag} {name}: marked claimable but signs {sorted(signs)} / CI {c['ci95']}")
        # 16. a pooled mean must lie inside its own seed range
        for name, c in rec["contrasts"].items():
            v = list(c["by_seed"].values())
            check(min(v) - 1e-9 <= c["pooled_mean"] <= max(v) + 1e-9,
                  f"{tag} {name}: pooled {c['pooled_mean']:.4f} outside seed range [{min(v):.4f}, {max(v):.4f}]")
        check(len(rec["seeds"]) >= 2, f"{tag}: only {len(rec['seeds'])} demonstration seed(s)", warn=True)
    # 17. the sweep must cover every model that actually has runs at this level
    have_runs = {m for m in load_config("models.yaml")["models"]
                 if (OUT_DIR / "runs" / m / f"L{level}" / "cot").is_dir()}
    in_sweep = {k.split("/")[0] for k in sw}
    check(not (have_runs - in_sweep),
          f"sweep L{level}: {sorted(have_runs - in_sweep)} have runs but are missing from the sweep "
          f"(re-run 56_seed_sweep.py; Table 1 and Figure 1 read this file)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()
    models = a.models or list(load_config("models.yaml")["models"])
    for L in a.levels:
        check_dataset(L)
        check_dataset(L, "_ident")
        check_sweeps(L)
        for m in models:
            check_runs(m, L)
            check_runs(m, L, "_ident")
            check_probes(m, L)
    print(f"checks complete: {len(FAIL)} failures, {len(WARN)} warnings\n")
    for w in WARN:
        print("  WARN ", w)
    for f in FAIL:
        print("  FAIL ", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
