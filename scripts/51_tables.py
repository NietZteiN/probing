#!/usr/bin/env python
"""results/summary -> paper/tables/*.tex and paper/numbers.tex. No number in the paper is typed
by hand: main.tex \\input{}s these, and any \\NUM{} key still undefined renders in red.

    python scripts/51_tables.py [--level 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR, PROJECT_ROOT, RESULTS_DIR, load_config  # noqa: E402

PAPER = PROJECT_ROOT / "paper"
PRETTY = {"llama32-3b": "Llama-3.2-3B", "llama32-3b-it": "Llama-3.2-3B-Inst.", "llama31-8b": "Llama-3.1-8B",
          "llama31-8b-it": "Llama-3.1-8B-Inst.", "gemma3-4b": "Gemma-3-4B", "olmo2-7b-it": "OLMo-2-7B-Inst.",
          "olmo3-7b-think": "OLMo-3-7B-Think"}


def pct(ci):
    if ci is None:
        return "--"
    p, lo, hi = ci
    return f"{100*p:.1f} {{\\scriptsize[{100*lo:.1f}, {100*hi:.1f}]}}"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--level", type=int, default=3); a = ap.parse_args()
    numbers = {}
    for k in load_config("models.yaml")["models"]:
        for regime in ("cot", "direct"):
            f = RESULTS_DIR / "summary" / k / f"L{a.level}" / regime / "behavior_table.json"
            if not f.exists():
                continue
            t = json.loads(f.read_text())
            for target in ("v2", "v1"):
                inc = t.get(f"incongruent@{target}"); cong = t.get(f"congruent@{target}"); neu = t.get("neutral")
                if not inc:
                    continue
                tag = f"{k.replace('-', '')}{regime}{target}"
                numbers[f"interference{tag}"] = f"{100*t[f'interference@{target}'][0]:.1f}"
                numbers[f"lurerate{tag}"] = f"{100*inc['lure_rate'][0]:.1f}"
    # amendment 2: pooled contrasts across demonstration seeds, when the sweep exists
    sweep = RESULTS_DIR / "summary" / f"seed_sweep_L{a.level}.json"
    if sweep.exists():
        sw = json.loads(sweep.read_text())
        for key, rec in sw.items():
            k, regime = key.split("/")
            tag = f"{k.replace('-', '')}{regime}"
            for name, c in rec["contrasts"].items():
                cname, target = name.split("@")
                numbers[f"{cname}{tag}{target}"] = f"{100*c['pooled_mean']:+.1f}"
                numbers[f"{cname}{tag}{target}lo"] = f"{100*c['ci95'][0]:+.1f}"
                numbers[f"{cname}{tag}{target}hi"] = f"{100*c['ci95'][1]:+.1f}"
            for g, v in rec["groups"].items():
                numbers[f"acc{tag}{g.replace('@', '')}"] = f"{100*v['acc_mean']:.1f}"
                numbers[f"acc{tag}{g.replace('@', '')}lo"] = f"{100*v['acc_range'][0]:.1f}"
                numbers[f"acc{tag}{g.replace('@', '')}hi"] = f"{100*v['acc_range'][1]:.1f}"
            numbers[f"nseeds{tag}"] = str(len(rec["seeds"]))
    (PAPER / "tables").mkdir(exist_ok=True)
    # ---- Table 1 (body): level 3 only, six columns. Other levels go to the appendix table.
    from cueconf.display import MODEL

    def build(levels, path, caption_levels):
        rows = []
        for L in levels:
            sp = RESULTS_DIR / "summary" / f"seed_sweep_L{L}.json"
            if not sp.exists():
                continue
            sw = json.loads(sp.read_text())
            # Table 1 is the PANEL: family and size. The tuned and reasoning rungs vary how one
            # spine was tuned and belong to the ladder in Appendix~\ref{app:kind}, not here.
            panel = {m for m, e in load_config("models.yaml")["models"].items()
                     if e.get("kind") not in ("tuned", "reasoning")}
            for key, rec in sorted(sw.items()):
                k, regime = key.split("/")
                # cot and direct are the paper's two regimes. `simple` (the value-only chain,
                # Appendix~\ref{app:simple}) is also in the sweep and must never render here: the
                # row label below collapses anything that is not cot to "direct".
                if k not in panel or regime not in ("cot", "direct"):
                    continue
                g, cons = rec["groups"], rec["contrasts"]

                def num(name):
                    c = cons.get(name)
                    if not c:
                        return "--"
                    return f"{100*c['pooled_mean']:+.1f}" + ("$^{*}$" if c["claimable"] else "")
                acc = f"{100*g['neutral']['acc_mean']:.0f}" if "neutral" in g else "--"
                rng = (f"\\,{{\\scriptsize[{100*g['neutral']['acc_range'][0]:.0f},{100*g['neutral']['acc_range'][1]:.0f}]}}"
                       if "neutral" in g else "")
                rows.append([MODEL.get(k, k)] + ([str(L)] if len(levels) > 1 else [])
                            + ["CoT" if regime == "cot" else "direct", acc + rng,
                               num("facilitation@v1"), num("lure_excess@v1"),
                               num("facilitation@v2"), num("lure_excess@v2")])
        if not rows:
            return 0
        ncol = len(rows[0])
        head_lvl = "Level & " if len(levels) > 1 else ""
        with path.open("w") as f:
            f.write("\\begin{tabular}{l" + ("c" if len(levels) > 1 else "") + "lc" + "r" * 4 + "}\n\\toprule\n")
            f.write(f"Model & {head_lvl}Regime & Neutral & \\multicolumn{{2}}{{c}}{{number word on the}} & "
                    f"\\multicolumn{{2}}{{c}}{{number word on the}} \\\\\n")
            f.write(f"& {'& ' if len(levels) > 1 else ''}& accuracy & \\multicolumn{{2}}{{c}}{{\\emph{{queried}} variable}} & "
                    f"\\multicolumn{{2}}{{c}}{{\\emph{{intermediate}} variable}} \\\\\n")
            f.write(f"\\cmidrule(lr){{{ncol-3}-{ncol-2}}}\\cmidrule(lr){{{ncol-1}-{ncol}}}\n")
            f.write(f"& {'& ' if len(levels) > 1 else ''}& (\\%) & helps & lures & helps & lures \\\\\n\\midrule\n")
            prev = None
            for r in rows:
                if prev is not None and r[0] != prev:
                    f.write("\\addlinespace\n")
                prev = r[0]
                f.write(" & ".join(r) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        return len(rows)

    # ---- narrative numbers quoted in the prose, computed here so none is hand-typed
    def narrative(numbers: dict) -> None:
        sweeps = {L: json.loads((RESULTS_DIR / "summary" / f"seed_sweep_L{L}.json").read_text())
                  for L in (1, 2, 3, 4, 5) if (RESULTS_DIR / "summary" / f"seed_sweep_L{L}.json").exists()}
        def scan(kind, regime):
            return [c["pooled_mean"] for sw in sweeps.values() for k, r in sw.items()
                    for n, c in r["contrasts"].items() if n.startswith(kind) and f"/{regime}" in k]
        if sweeps:
            numbers["facilitationmax"] = f"{100*max(scan('facilitation', 'direct')):.0f}"
            numbers["lureexcessmax"] = f"{100*max(scan('lure_excess', 'direct')):.0f}"
            numbers["chain-lure-bound"] = f"{100*max(abs(v) for v in scan('lure_excess', 'cot')):.1f}"
            numbers["n-models"] = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                                   8: "eight"}.get(len({k.split('/')[0] for sw in sweeps.values() for k in sw}), "several")
            numbers["model-list"] = ", ".join(sorted({MODEL.get(k.split("/")[0], k.split("/")[0])
                                                      for sw in sweeps.values() for k in sw}))
        # equivalence bounds over every chain cell (E37): counts, and the exception(s) named
        eqf = RESULTS_DIR / "summary" / "equivalence.json"
        if eqf.exists():
            eq = json.loads(eqf.read_text())
            cot = {k: v for k, v in eq["cells"].items() if "/cot/" in k}
            direct = {k: v for k, v in eq["cells"].items() if "/direct/" in k}
            numbers["eq-delta"] = f"{100*eq['delta']:.0f}"
            numbers["eq-cot-cells"] = str(len(cot))
            numbers["eq-cot-equivalent"] = str(sum(v["equivalent"] for v in cot.values()))
            numbers["eq-direct-cells"] = str(len(direct))
            numbers["eq-direct-effect"] = str(sum(v["claimable"] for v in direct.values()))
            exc = [(k, v) for k, v in cot.items() if not v["equivalent"]]
            numbers["eq-cot-exceptions"] = str(len(exc))
            if exc:
                k, v = max(exc, key=lambda kv: abs(kv[1]["mean"]))
                L, m, _, c = k.split("/")
                numbers["eq-exc-model"] = MODEL.get(m, m)
                numbers["eq-exc-level"] = L[1:]
                numbers["eq-exc-role"] = "queried" if c.endswith("v1") else "intermediate"
                numbers["eq-exc-mean"] = f"{100*v['mean']:+.1f}"
                numbers["eq-exc-lo"] = f"{100*v['ci95'][0]:+.1f}"
                numbers["eq-exc-hi"] = f"{100*v['ci95'][1]:+.1f}"
                sw = sweeps.get(int(L[1:]), {}).get(f"{m}/cot", {}).get("groups", {})
                if sw:
                    numbers["eq-exc-neutral-acc"] = f"{100*sw['neutral']['acc_mean']:.0f}"
                    numbers["eq-exc-letter-acc"] = f"{100*sw['letter']['acc_mean']:.0f}"
        # the level-4 word-class cost on the queried variable (E26), pooled over seeds, both signs
        sw4 = sweeps.get(4, {}).get("llama32-3b/cot")
        if sw4:
            g = sw4["groups"]
            cost = g["neutral"]["acc_mean"] - 0.5 * (g["congruent@v1"]["acc_mean"] + g["incongruent@v1"]["acc_mean"])
            numbers["wordclass-cost-L4"] = f"{100*cost:.1f}"
        # value-written control (62_value_written.py): does the chain protect BECAUSE it writes
        # the value? Pooled excess over the matched neutral twin put through the same split.
        vwf = RESULTS_DIR / "summary" / "value_written.json"
        if vwf.exists():
            import numpy as np
            vw = json.loads(vwf.read_text())
            for tag in ("wrote", "not"):
                sel = [r for r in vw if r[f"n_{tag}"] >= 50 and r[f"excess_{tag}"] is not None]
                if sel:
                    w = [r[f"n_{tag}"] for r in sel]
                    numbers[f"vw-excess-{tag}"] = f"{100*np.average([r[f'excess_{tag}'] for r in sel], weights=w):+.2f}"
                    numbers[f"vw-cells-{tag}"] = str(len(sel))
            olmo = [r for r in vw if r["model"] == "olmo2-1b-it" and r["target"] == "v2"
                    and r["regime"] == "cot" and r.get("excess_wrote_ci")]
            if olmo:
                r = olmo[0]; ci = r["excess_wrote_ci"]
                numbers["vw-olmo-excess"] = f"{100*r['excess_wrote']:+.1f}"
                numbers["vw-olmo-lo"] = f"{100*ci[0]:+.1f}"
                numbers["vw-olmo-hi"] = f"{100*ci[1]:+.1f}"
                numbers["vw-olmo-n"] = str(r["n_wrote"])
        # own-chain probe figure (26_own_chain_probe.py / 67_own_chain_figure.py): how many instances it shows
        oc = RESULTS_DIR / "summary" / "llama32-3b" / "L3" / "cot" / "own_chain_v1.json"
        ob = OUT_DIR / "runs" / "llama32-3b" / "L3" / "cot" / "incongruent@v1" / "behavior.jsonl"
        if oc.exists() and ob.exists():
            numbers["ownchain-n"] = str(len(json.loads(oc.read_text())))
            numbers["ownchain-total"] = str(sum(1 for l in ob.open() if not json.loads(l)["correct"]))
        kt = RESULTS_DIR / "summary" / "llama32-3b" / "L3" / "cot" / "kudo_table.json"
        if kt.exists():
            k = json.loads(kt.read_text())
            for v in ("v1", "v2"):
                c = k.get(f"train_letter/{v}", {}).get("letter")
                if c:
                    numbers[f"replication-pre-{v}"] = f"{100*c['acc_pre_cot']:.1f}"
                    numbers[f"replication-eq-{v}"] = c["t_star_segment"].split(":")[1]
        # level-4 lure mass at the ANSWER position for both the bound (incongruent@v2) and the
        # unbound (irrelevant@v3) number word, best-accuracy layer, so the two are comparable
        g4f = RESULTS_DIR / "summary" / "llama32-3b" / "L4" / "cot" / "probes_grid.json"
        if g4f.exists():
            g4 = json.loads(g4f.read_text())
            for role, cond, key in (("v2", "incongruent@v2", "bound-mass"), ("v3", "irrelevant@v3", "unbound-mass")):
                cells = [v for kk, v in g4.get(f"train_neutral/{role}", {}).items() if kk.startswith("anspre|") and cond in v]
                if cells:
                    best = max(cells, key=lambda v: v["neutral"]["accuracy"][0])
                    numbers[key] = f"{best[cond]['lure_mass'][0]:.3f}"
                    if role == "v3":
                        numbers["distractor-acc"] = f"{best['neutral']['accuracy'][0]:.2f}"
        # where the removal grid puts the effect: segment and layer windows with the highest removal
        ptg = RESULTS_DIR / "summary" / "llama32-3b" / "L3" / "direct" / "patching.json"
        if ptg.exists():
            grid = json.loads(ptg.read_text()).get("grid_neutral@v1", {})
            rows = [(kk.split("|")[0], kk.split("|")[1], v["anspre"]["lure_removed"]) for kk, v in grid.items() if "anspre" in v]
            if rows:
                seg, win, top = max(rows, key=lambda r: r[2])
                good = sorted({int(w[1:].split("-")[0]) for sg, w, r in rows if sg == seg and r >= 0.5} | {int(w[1:].split("-")[1]) for sg, w, r in rows if sg == seg and r >= 0.5})
                numbers["grid-layers"] = f"{good[0]}--{good[-1]}" if good else win[1:]
                numbers["grid-top-removed"] = f"{100*top:.0f}\\%"
        # per-demonstration-set ranges behind two pooled numbers the text quotes
        def by_seed_range(L, key, contrast):
            c = sweeps.get(L, {}).get(key, {}).get("contrasts", {}).get(contrast)
            if c and c.get("by_seed"):
                vals = [100*x for x in c["by_seed"].values()]
                return f"{min(vals):.0f}", f"{max(vals):.0f}"
            return None
        r = by_seed_range(5, "llama31-8b/direct", "lure_excess@v1")
        if r:
            numbers["lureexcessmax-lo"], numbers["lureexcessmax-hi"] = r
        sw4 = sweeps.get(4, {}).get("llama32-3b/cot")
        if sw4 and "acc_by_seed" in sw4["groups"]["neutral"]:
            g = sw4["groups"]; costs = []
            for sd in g["neutral"]["acc_by_seed"]:
                costs.append(100*(g["neutral"]["acc_by_seed"][sd] - 0.5*(g["congruent@v1"]["acc_by_seed"][sd] + g["incongruent@v1"]["acc_by_seed"][sd])))
            numbers["wordclass-cost-L4-lo"], numbers["wordclass-cost-L4-hi"] = f"{min(costs):.0f}", f"{max(costs):.0f}"
        # how many models have every level
        if sweeps:
            per = {}
            for L, sw in sweeps.items():
                for kk in sw: per.setdefault(kk.split("/")[0], set()).add(L)
            numbers["n-models-all-levels"] = {1: "one", 2: "two", 3: "three"}.get(sum(1 for v in per.values() if len(v) == 5), str(sum(1 for v in per.values() if len(v) == 5)))
        # chain regime: how often injecting the name's activations changes the answer (max over
        # layers and models with patching), and the most lure errors any model made in 2,000
# injection under CoT: the exception model is reported separately from the rest, because it is
        # the one model where the name still moves the answer through a forced correct chain
        EXC = "olmo2-1b-it"
        inj, err, exc = [], [], {}
        for pf in (RESULTS_DIR / "summary").glob("*/L3/cot/patching.json"):
            d = json.loads(pf.read_text()); mk = pf.parts[-4]
            for kk in ("inject@v1", "inject@v2"):
                for L, v in d.get(kk, {}).items():
                    a_ = v.get("anspre") if isinstance(v, dict) else None
                    if a_ and a_.get("lure_injected") is not None:
                        (exc.setdefault(kk, []) if mk == EXC else inj).append((a_["lure_injected"], L, a_.get("damage")))
            for kk in ("main@v1", "main@v2"):
                a_ = d.get(kk, {}).get("ALL", {}).get("anspre")
                if a_: err.append(a_["n_lure_err"])
        if inj:
            numbers["inject-cot-max"] = f"{100*max(v for v, _, _ in inj):.1f}\\%"
        if err:
            numbers["cot-lure-err-max"] = str(max(err))
        # the exception model, read straight from its patching files (per-layer summaries)
        pdir = OUT_DIR / "patching" / EXC / "L3" / "cot"
        if pdir.exists():
            best = None
            for kk in ("inject@v1", "inject@v2"):
                f_ = pdir / f"{kk}.json"
                if not f_.exists(): continue
                sm = json.loads(f_.read_text())["summary"]
                for L, v in sm.items():
                    a_ = v.get("anspre") if isinstance(v, dict) else None
                    if a_ and a_.get("lure_injected") is not None and (best is None or a_["lure_injected"] > best[0]):
                        best = (a_["lure_injected"], L, a_["damage"], kk)
            if best:
                v, L, dmg, kk = best
                numbers["exc-inject"] = f"{100*v:.1f}\\%"
                numbers["exc-inject-layer"] = L.lstrip("L")
                numbers["exc-inject-damage"] = f"{100*dmg:.1f}\\%"
                # the word control at the SAME layer; fall back to the other target's control
                for cand in (kk.replace("inject", "ctl_word"), "ctl_word@v1.json", "ctl_word@v2.json"):
                    cf = pdir / (cand if cand.endswith(".json") else cand + ".json")
                    if cf.exists():
                        c = json.loads(cf.read_text())["summary"].get(L, {}).get("anspre")
                        if c:
                            numbers["exc-ctlword"] = f"{100*c['damage']:.1f}\\%"
                            break
# positional-copy control (E22): among CoT lure errors at the answer, how often the number
        # standing just before the answer is the lure rather than the true value
        cc_err = cc_lure = cc_true = 0; cc_cells = 0
        for cf in (RESULTS_DIR / "summary").glob("*/L*/cot*/copy_control.json"):
            for v in json.loads(cf.read_text()).values():
                n = v.get("n_lure_errors_at_p5") or 0
                if n:
                    cc_cells += 1; cc_err += n
                    cc_lure += n * (v.get("trailing_is_lure") or 0.0)
                    cc_true += n * (v.get("trailing_is_true_intermediate") or 0.0)
        if cc_err:
            numbers["copy-n-err"] = str(cc_err)
            numbers["copy-lure-frac"] = f"{100*cc_lure/cc_err:.0f}\\%"
            numbers["copy-true-frac"] = f"{100*cc_true/cc_err:.0f}\\%"
# Is the target's value linearly decodable at the step where the chain writes it, and does
        # that line up with the behavioural lure excess? One row per (model with probes, role).
        vs = []
        for gf in (RESULTS_DIR / "summary").glob("*/L3/cot/probes_grid.json"):
            mk = gf.parts[-4]; gg = json.loads(gf.read_text())
            for role in ("v1", "v2"):
                cells = {k.split("|")[1]: v for k, v in gg.get(f"train_neutral/{role}", {}).items()
                         if k.startswith(f"cotpre@{role}|")}
                if not cells: continue
                L = max(cells, key=lambda L: cells[L]["neutral"]["accuracy"][0] - cells[L]["neutral"]["control_acc"])
                acc = cells[L]["neutral"]["accuracy"][0]
                ex = sweeps.get(3, {}).get(f"{mk}/cot", {}).get("contrasts", {}).get(f"lure_excess@{role}")
                if ex: vs.append((mk, role, acc, ex["pooled_mean"], ex["claimable"]))
        if vs:
            hi = [r for r in vs if r[2] >= 0.9]; lo = [r for r in vs if r[2] < 0.9]
            numbers["vs-cells"] = str(len(vs))
            numbers["vs-decodable"] = str(len(hi))
            numbers["vs-decodable-min"] = f"{min(r[2] for r in hi):.2f}" if hi else "--"
            numbers["vs-decodable-maxexcess"] = f"{100*max(abs(r[3]) for r in hi):.1f}" if hi else "--"
            if lo:
                r = max(lo, key=lambda r: abs(r[3]))
                numbers["vs-undecodable-acc"] = f"{r[2]:.2f}"
                numbers["vs-undecodable-n"] = str(len(lo))
# identifier-name scheme (E38): a digit inside an identifier (q4) instead of a number word
        idf = RESULTS_DIR / "summary" / "seed_sweep_L3_ident.json"
        if idf.exists():
            idn = json.loads(idf.read_text()); wrd = sweeps.get(3, {})
            def span(d, regime, contrast):
                v = [c["pooled_mean"] for k, c in ((k, d[k]["contrasts"].get(contrast)) for k in d)
                     if k.endswith("/" + regime) and c]
                cl = [k for k in d if k.endswith("/" + regime) and (d[k]["contrasts"].get(contrast) or {}).get("claimable")]
                return (min(v), max(v), len(cl), len(v)) if v else None
            common = {k for k in set(idn) & set(wrd)}
            for tag, d in (("ident", {k: v for k, v in idn.items() if k in common}),
                           ("word", {k: v for k, v in wrd.items() if k in common})):
                sp = span(d, "direct", "lure_excess@v1")
                if sp:
                    numbers[f"{tag}-direct-lure-lo"] = f"{100*sp[0]:+.1f}"
                    numbers[f"{tag}-direct-lure-hi"] = f"{100*sp[1]:+.1f}"
                    numbers[f"{tag}-direct-lure-claim"] = str(sp[2])
                    numbers[f"{tag}-direct-cells"] = str(sp[3])
            numbers["ident-models"] = {2: "two", 3: "three", 4: "four"}.get(len({k.split("/")[0] for k in common}), str(len({k.split("/")[0] for k in common})))
        irf = RESULTS_DIR / "summary" / "irrelevant_L4.json"
        if irf.exists():
            ir = json.loads(irf.read_text())
            for base in ("cot", "direct"):
                cells = {k: v for k, v in ir.items() if k.endswith("/" + base)}
                if cells:
                    lo_k = min(cells, key=lambda k: cells[k]["lure_excess"]); hi_k = max(cells, key=lambda k: cells[k]["lure_excess"])
                    numbers[f"irr-{base}-lo"] = f"{100*cells[lo_k]['lure_excess']:+.1f}"
                    numbers[f"irr-{base}-hi"] = f"{100*cells[hi_k]['lure_excess']:+.1f}"
                    numbers[f"irr-{base}-claimable"] = str(sum(v["claimable"] for v in cells.values()))
                    numbers[f"irr-{base}-cells"] = str(len(cells))
            numbers["irr-acc-delta-max"] = f"{100*max(abs(v['acc_delta']) for v in ir.values()):.1f}"
        rep = RESULTS_DIR / "summary" / "llama32-3b" / "L3" / "cot" / "replication.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            for v in ("v1", "v2"):
                numbers.setdefault(f"replication-pre-{v}", f"{100*r[v]['acc_pre_cot']:.1f}")
        pg = RESULTS_DIR / "summary" / "llama32-3b" / "L3" / "cot" / "probes_grid.json"
        if pg.exists():
            grid = json.loads(pg.read_text())["train_neutral/v2"]
            def best(pos):
                cells = [v for k, v in grid.items() if k.split("|")[0] == pos and "incongruent@v2" in v]
                return max(cells, key=lambda v: v["neutral"]["accuracy"][0]) if cells else None
            b2, b3, b4 = best("end@v2"), best("query"), best("cotpre@v2")
            if b2:
                numbers["p2-acc"] = f"{b2['neutral']['accuracy'][0]:.2f}"
                numbers["p2-sel"] = f"{b2['neutral']['accuracy'][0] - b2['neutral']['control_acc']:.2f}"
                numbers["p2-lure"] = f"{b2['incongruent@v2']['lure_rate'][0]:.2f}"
            if b3:
                numbers["p3-lure"] = f"{100*b3['incongruent@v2']['lure_rate'][0]:.0f}\\%"
                numbers["p3-ctl"] = f"{b3['neutral']['control_acc']:.2f}"
            if b4:
                numbers["p4-acc"] = f"{b4['neutral']['accuracy'][0]:.2f}"
                numbers.setdefault("bound-mass", f"{b4['incongruent@v2']['lure_mass'][0]:.3f}")  # L4 anspre value set above wins
        g4 = RESULTS_DIR / "summary" / "llama32-3b" / "L4" / "cot" / "probes_grid.json"
        if g4.exists():
            gg = json.loads(g4.read_text()).get("train_neutral/v3", {})
            cells = [v for k, v in gg.items() if k.split("|")[0] == "anspre" and "irrelevant@v3" in v]
            if cells:
                numbers["unbound-mass"] = f"{max(cells, key=lambda v: v['neutral']['accuracy'][0])['irrelevant@v3']['lure_mass'][0]:.3f}"
        pt = RESULTS_DIR / "summary" / "llama32-3b" / "L3" / "direct" / "patching.json"
        if pt.exists():
            d = json.loads(pt.read_text())
            m, w, l = (d[f"{c}@v1"]["ALL"]["anspre"] for c in ("main", "ctl_word", "ctl_lure"))
            numbers["lure-removed-3b"] = f"{100*m['lure_removed']:.0f}\\%"
            numbers["nld-3b"] = f"{m['normalized_ld_mean']:.2f}"
            numbers["control-damage"] = f"{100*w['damage']:.0f}\\%"
            numbers["follow-new-lure"] = f"{100*l['follows_src_lure']:.0f}\\%"
        numbers.setdefault("grid-layers", "?")

    # ---- E21 + E7: the two robustness statistics (scripts/58_robust_stats.py)
    rs = RESULTS_DIR / "summary" / "robust_stats_L3.json"
    if rs.exists():
        d = json.loads(rs.read_text())
        numbers["mix-cells"] = str(d["mixed_fitted"])
        numbers["mix-agree"] = str(d["mixed_agree"])
        numbers["mix-unfitted"] = str(d["mixed_cells"] - d["mixed_fitted"])
        numbers["mix-cap"] = str(d.get("max_sets", 0))
        fitted = [(k, c) for k, c in d["mixed_effects"].items() if "coef" in c.get("mixed", {})]
        diff = [c for k, c in fitted if c.get("verdict") == "differ"]
        numbers["mix-differ-n"] = str(len(diff))
        # a disagreement matters only if it removes support for something the body claims
        numbers["mix-lost"] = str(sum(1 for _, c in fitted
                                      if (c["boot_lo"] > 0 or c["boot_hi"] < 0) and not c["mixed"]["excludes_zero"]))
        numbers["mix-differ-neg"] = str(sum(1 for c in diff if c["mixed"]["coef"] < 0))
        lk = d["link"]
        numbers["link-fits"] = str(lk["total_fits"]); numbers["link-sig"] = str(lk["total_sig"])
        numbers["link-neg"] = str(lk["sig_negative"]); numbers["link-pos"] = str(lk["sig_positive"])
        import collections as _c
        pos_models = _c.Counter(r["model"] for r in lk["fits"] if r["p"] < 0.05 and r["coef"] > 0)
        if pos_models:
            numbers["link-pos-topmodel-n"] = str(pos_models.most_common(1)[0][1])

    # ---- E17 probe recipe + E18 patching scope (69_recipe_scope.py)
    rsf = RESULTS_DIR / "summary" / "recipe_scope_L3.json"
    if rsf.exists():
        rs = json.loads(rsf.read_text())
        rec = rs.get("recipe", {})
        for name in ("sgd", "lbfgs", "std"):
            for pos, key in (("v1/cotpre@v1", "cotpre-v1"), ("v2/cotpre@v2", "cotpre-v2"),
                             ("v1/end@v1", "end-v1"), ("v2/end@v2", "end-v2"),
                             ("v1/query", "query-v1"), ("v2/query", "query-v2"), ("v1/anspre", "anspre")):
                v = rec.get(name, {}).get(pos)
                if v is not None:
                    numbers[f"e17-{name}-{key}"] = f"{v:.3f}"
        vals = [v for name in rec for k, v in rec[name].items() if k.endswith(("cotpre@v1", "cotpre@v2"))]
        if vals:
            numbers["e17-value-lo"] = f"{min(vals):.2f}"; numbers["e17-value-hi"] = f"{max(vals):.2f}"
        ends = [v for name in rec for k, v in rec[name].items() if "end@" in k]
        if ends:
            numbers["e17-end-lo"] = f"{min(ends):.2f}"; numbers["e17-end-hi"] = f"{max(ends):.2f}"
        qs = [v for name in rec for k, v in rec[name].items() if k.endswith("query")]
        if qs:
            numbers["e17-query-lo"] = f"{min(qs):.2f}"; numbers["e17-query-hi"] = f"{max(qs):.2f}"
        sc = rs.get("scope", {})
        for role in ("v1", "v2"):
            for tag in ("all", "prompt"):
                cells = sc.get(f"{role}/{tag}", {})
                wins = {k: v for k, v in cells.items() if k.startswith("W")}
                if wins:
                    bk, bv = max(wins.items(), key=lambda kv: kv[1]["lure_removed"])
                    numbers[f"e18-{role}-{tag}-best"] = bk.replace("W", "").replace("-", "--")
                    numbers[f"e18-{role}-{tag}-removed"] = f"{100*bv['lure_removed']:.0f}"
                    numbers[f"e18-{role}-{tag}-damage"] = f"{100*bv['damage']:.0f}"
                w03 = cells.get("W0-3")
                if w03:
                    numbers[f"e18-{role}-{tag}-w03"] = f"{100*w03['lure_removed']:.0f}"
                    numbers[f"e18-{role}-{tag}-w03damage"] = f"{100*w03['damage']:.0f}"
        lines = ["\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
                 "Position & Variable & SGD (reported) & L-BFGS & standardized \\\\", "\\midrule"]
        for pos, lab, role in (("end@{r}", "defining equation", "v1"), ("end@{r}", "defining equation", "v2"),
                               ("query", "query", "v1"), ("query", "query", "v2"),
                               ("cotpre@{r}", "value step", "v1"), ("cotpre@{r}", "value step", "v2"),
                               ("anspre", "answer", "v1")):
            key = f"{role}/{pos.format(r=role)}"
            row = [rec.get(n, {}).get(key) for n in ("sgd", "lbfgs", "std")]
            if any(v is not None for v in row):
                var = "queried" if role == "v1" else "intermediate"
                lines.append(f"{lab} & {var} & " + " & ".join(f"{v:.3f}" if v is not None else "--" for v in row) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (PAPER / "tables" / "recipe.tex").write_text("\n".join(lines) + "\n")

    # ---- E31 value-only chain (65_simple_chain.py)
    scf = RESULTS_DIR / "summary" / "simple_chain_L3.json"
    if scf.exists():
        sc = json.loads(scf.read_text())
        SH = {"llama32-3b": "l3b", "llama31-8b": "l8b", "olmo2-1b-it": "olmo1"}
        for m, sh in SH.items():
            for role in ("v1", "v2"):
                c = sc.get(f"{m}/L3/simple/{role}")
                if not c:
                    continue
                w = c["written_lure_excess"]
                numbers[f"simple-{sh}-{role}"] = f"{w['mean']:+.2f}"
                numbers[f"simple-{sh}-{role}-lo"] = f"{w['lo']:+.2f}"; numbers[f"simple-{sh}-{role}-hi"] = f"{w['hi']:+.2f}"
                numbers[f"simple-{sh}-{role}-seeds"] = " / ".join(f"{v:+.1f}" for v in w["by_seed"].values())
                numbers[f"simple-{sh}-acc"] = f"{c['acc_neutral']:.0f}"
        # probe decodability at the step that writes the value, simple vs the full chain
        import glob as _glob
        for m, sh in SH.items():
            for reg in ("simple", "cot"):
                for role in ("v1", "v2"):
                    f = OUT_DIR / "probes" / m / "L3" / reg / "train_neutral" / f"{role}.json"
                    if not f.exists():
                        continue
                    best = 0.0
                    for r in json.loads(f.read_text())["results"]:
                        if r["position"] != f"cotpre@{role}":
                            continue
                        acc = r["eval"].get("neutral", {}).get("accuracy")
                        if acc and acc > best:
                            best = acc
                    if best:
                        numbers[f"simpleprobe-{sh}-{reg}-{role}"] = f"{best:.2f}"

    # ---- E40 model-kind ladder (68_model_kind.py)
    mkf = RESULTS_DIR / "summary" / "model_kind_L3.json"
    if mkf.exists():
        mk = json.loads(mkf.read_text())
        RUNGS = [("llama31-8b", "base", "base"), ("llama31-8b-it", "instruct", "it"),
                 ("llama31-8b-it-ft", "single-task finetune", "ft"),
                 ("llama31-8b-it-ftmulti", "multi-task finetune", "ftm"),
                 ("llama31-8b-it-merged", "weight merge", "merge"),
                 ("nemotron-nano-8b", "reasoning-tuned", "reas")]
        for m, _, sh in RUNGS:
            for role in ("v1", "v2"):
                for reg, rtag in (("direct", ""), ("cot", "-cot")):
                    c = mk.get(f"{m}/L3/{reg}/{role}")
                    if not c or "lure_excess" not in c:
                        continue
                    e = c["lure_excess"]
                    numbers[f"kind-{sh}-{role}{rtag}"] = f"{e['mean']:+.2f}"
                    numbers[f"kind-{sh}-{role}{rtag}-lo"] = f"{e['lo']:+.2f}"
                    numbers[f"kind-{sh}-{role}{rtag}-hi"] = f"{e['hi']:+.2f}"
                    if reg == "direct":
                        numbers[f"kind-{sh}-acc"] = f"{c['acc_neutral']:.1f}"
                    else:
                        numbers[f"kind-{sh}-acc-cot"] = f"{c['acc_neutral']:.1f}"
        tuned = [f"{m}/L3/direct/{r}" for m, _, sh in RUNGS if sh in ("ft", "ftm", "merge") for r in ("v1", "v2")]
        it = {r: mk.get(f"llama31-8b-it/L3/direct/{r}", {}).get("lure_excess", {}).get("mean") for r in ("v1", "v2")}
        gaps = [abs(mk[k]["lure_excess"]["mean"] - it[k.rsplit("/", 1)[1]]) for k in tuned
                if k in mk and it.get(k.rsplit("/", 1)[1]) is not None]
        if gaps:
            numbers["kind-maxgap"] = f"{max(gaps):.2f}"; numbers["kind-tuned-cells"] = str(len(gaps))
        cotvals = [abs(mk[f"{m}/L3/cot/{r}"]["lure_excess"]["mean"]) for m, _, _ in RUNGS for r in ("v1", "v2")
                   if f"{m}/L3/cot/{r}" in mk]
        if cotvals:
            numbers["kind-cot-max"] = f"{max(cotvals):.2f}"
        numbers["kind-rungs"] = str(len(RUNGS))
        lines = ["\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
                 "Tuning & Model & Queried & Intermediate & Acc. \\\\", "\\midrule"]
        DISP = {"llama31-8b": "Llama-3.1-8B", "llama31-8b-it": "\\quad + instruct",
                "llama31-8b-it-ft": "\\quad + task LoRA", "llama31-8b-it-ftmulti": "\\quad + multi-task LoRA",
                "llama31-8b-it-merged": "\\quad + TIES merge", "nemotron-nano-8b": "Nemotron-Nano-8B"}
        for m, lab, _ in RUNGS:
            cells = []
            for role in ("v1", "v2"):
                c = mk.get(f"{m}/L3/direct/{role}")
                if not c or "lure_excess" not in c:
                    cells.append("--"); continue
                e = c["lure_excess"]
                cells.append(f"{e['mean']:+.2f}{'$^{*}$' if e['claimable'] else ''}")
            acc = mk.get(f"{m}/L3/direct/v1", {}).get("acc_neutral")
            if acc is None:
                acc = mk.get(f"{m}/L3/direct/v2", {}).get("acc_neutral")
            if acc is None:
                continue
            lines.append(f"{lab} & {DISP[m]} & " + " & ".join(cells) + f" & {acc:.1f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (PAPER / "tables" / "modelkind.tex").write_text("\n".join(lines) + "\n")

    # ---- E23: Geirhos-style lure index, lure / (lure + true), congruent trials excluded.
    # Reported as a range beside the twin-subtracted excess, which the matched design makes the
    # stronger quantity; the index is the comparable number for the shape/texture literature.
    import numpy as _np
    idx = {"direct": [], "cot": []}
    for m in load_config("models.yaml")["models"]:
        if load_config("models.yaml")["models"][m].get("kind") in ("tuned", "reasoning"):
            continue
        for reg in ("direct", "cot"):
            f = RESULTS_DIR / "summary" / m / "L3" / reg / "behavior_table.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            for g, rec in d.items():
                if g.startswith("incongruent@") and rec.get("lure_index") is not None:
                    v = rec["lure_index"]
                    idx[reg].append(v[0] if isinstance(v, list) else v)
    for reg in ("direct", "cot"):
        if idx[reg]:
            numbers[f"lureindex-{reg}-lo"] = f"{min(idx[reg]):.2f}"
            numbers[f"lureindex-{reg}-hi"] = f"{max(idx[reg]):.2f}"
            numbers[f"lureindex-{reg}-n"] = str(len(idx[reg]))

    narrative(numbers)
    n_main = build([3], PAPER / "tables" / "behavior.tex", "3")
    n_app = build([1, 2, 4, 5], PAPER / "tables" / "behavior_levels.tex", "1, 2, 4 and 5")
    print(f"Table 1: {n_main} rows (level 3); appendix table: {n_app} rows")
    with (PAPER / "numbers.tex").open("w") as f:
        f.write("% generated by scripts/51_tables.py -- do not edit\n")
        for key, val in sorted(numbers.items()):
            f.write(f"\\expandafter\\newcommand\\csname NUMval@{key}\\endcsname{{{val}}}\n")
        f.write("\\renewcommand{\\NUM}[1]{\\ifcsname NUMval@#1\\endcsname\\csname NUMval@#1\\endcsname\\else\\textcolor{red}{$\\langle\\langle$\\texttt{#1}$\\rangle\\rangle$}\\fi}\n")
    print(f"wrote {len(numbers)} numbers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
