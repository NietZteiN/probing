"""Aggregation of patching rows into rates. Torch-free so scripts/50_analysis.py can import it on
the login node (importing cueconf.patching pulls torch, which dies under the 8 GB cap)."""
from __future__ import annotations

import numpy as np


def summarize(rows: list[dict], sets) -> dict:
    out = {}
    for sname, _ in sets:
        agg = {}
        for lab in ("anspre", "cotpre@v1", "cotpre@v2", "cotpre@v3"):
            rs = [r for r in rows if lab in r["base"]]
            if not rs:
                continue
            lure_err = [r for r in rs if r["lure"] is not None and r["base"][lab] == str(r["lure"])]
            correct = [r for r in rs if r["base"][lab] == r["gold"][lab]]
            # E24: normalised logit difference, (LD_patched - LD_base) / (LD_src - LD_base); 1 = fully restored
            nld = []
            for r in rs:
                b, sref, pv = r.get("base_ld", {}).get(lab), r.get("src_ld", {}).get(lab), r.get("patched_ld", {}).get(sname, {}).get(lab)
                if b is not None and sref is not None and pv is not None and abs(sref - b) > 1e-6:
                    nld.append((pv - b) / (sref - b))
            # lure removed: among base lure errors, the patched answer is no longer the lure (does not
            # require it to be CORRECT, which the source twin itself may not be)
            lure_removed = (np.mean([r["patched"][sname][lab] != str(r["lure"]) for r in lure_err]) if lure_err else None)
            # matches the source twin's own answer (available for runs after 2026-09-12)
            with_src = [r for r in rs if "src_pred" in r]
            matches_src = (np.mean([r["patched"][sname][lab] == r["src_pred"][lab] for r in with_src]) if with_src else None)
            correct_before = [r for r in rs if r["base"][lab] == r["gold"][lab]]
            injected = (np.mean([r["patched"][sname][lab] == str(r["src_lure"]) for r in correct_before
                                 if r["src_lure"] is not None]) if any(r["src_lure"] is not None for r in correct_before) else None)
            agg[lab] = {
                "n": len(rs),
                "lure_injected": injected,
                "lure_removed": lure_removed,
                "matches_source": matches_src,
                "normalized_ld_mean": (float(np.mean(nld)) if nld else None),
                "normalized_ld_median": (float(np.median(nld)) if nld else None),
                "base_acc": np.mean([r["base"][lab] == r["gold"][lab] for r in rs]),
                "patched_acc": np.mean([r["patched"][sname][lab] == r["gold"][lab] for r in rs]),
                "n_lure_err": len(lure_err),
                "recovery": (np.mean([r["patched"][sname][lab] == r["gold"][lab] for r in lure_err]) if lure_err else None),
                "damage": (np.mean([r["patched"][sname][lab] != r["gold"][lab] for r in correct]) if correct else None),
                "follows_src_lure": (np.mean([r["patched"][sname][lab] == str(r["src_lure"]) for r in rs if r["src_lure"] is not None])
                                     if any(r["src_lure"] is not None for r in rs) else None),
            }
        out[sname] = agg
    return out




def summarize_grid(rows: list[dict]) -> dict:
    out = {}
    cells = sorted({c for r in rows for c in r["cells"]})
    for c in cells:
        agg = {}
        for lab in ("anspre", "cotpre@v1", "cotpre@v2", "cotpre@v3"):
            rs = [r for r in rows if lab in r["base"] and c in r["cells"]]
            if not rs:
                continue
            lure_err = [r for r in rs if r["lure"] is not None and r["base"][lab] == str(r["lure"])]
            agg[lab] = {
                "n": len(rs),
                "n_lure_err": len(lure_err),
                "success_to_source": float(np.mean([r["cells"][c]["pred"][lab] == r["src_gold"][lab] for r in rs])),
                "changed": float(np.mean([r["cells"][c]["pred"][lab] != r["base"][lab] for r in rs])),
                "lure_removed": (float(np.mean([r["cells"][c]["pred"][lab] != str(r["lure"]) for r in lure_err])) if lure_err else None),
                "follows_src_lure": (float(np.mean([r["cells"][c]["pred"][lab] == str(r["src_lure"]) for r in rs if r["src_lure"] is not None]))
                                     if any(r["src_lure"] is not None for r in rs) else None),
            }
        out[c] = agg
    return out
