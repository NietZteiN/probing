#!/usr/bin/env python
"""Collect the authorized probe reruns/extensions after the two day-long workers exit.

Incomplete experiments stay unchecked. Raw caches remain separate, and no measurement is
invented when a worker fails or runs out of time.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parent / "codecue"
sys.path.insert(0, str(ROOT / "src"))
from cueconf.config import OUT_DIR, RESULTS_DIR  # noqa: E402
from cueconf.display import MODEL  # noqa: E402

PANEL = ("llama32-3b-it", "llama31-8b-it", "gemma3-12b-it")
LOG = ROOT / "log/round4_2026-10-02"


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_probe(path, n_train, suffix):
    data = json.loads(path.read_text())
    train_dir = Path(data["train_dir"])
    train = json.loads((train_dir / "meta.json").read_text())
    if not train.get("pre_value_boundary_checked"):
        raise ValueError(f"{path}: pre-value token boundary was not checked")
    if train["n"] != n_train or len(train["instances"]) != n_train:
        raise ValueError(f"{path}: expected {n_train} training rows")
    if data["optimizer"] != "sgd" or data["epochs"] != 10000 or data["lr"] != .001:
        raise ValueError(f"{path}: changed training recipe")
    if data.get("standardize"):
        raise ValueError(f"{path}: expected the primary unstandardized recipe")
    train_ids = {x["set_id"] for x in train["instances"]}
    for group in ("neutral", "incongruent@v1", "incongruent@v2"):
        meta = json.loads((train_dir.parent / (group + suffix) / "meta.json").read_text())
        if not meta.get("pre_value_boundary_checked"):
            raise ValueError(f"{path}: unchecked pre-value boundary in {group}")
        expected = 1000 if "alltok" in suffix else 2000
        if meta["n"] != expected or len(meta["instances"]) != expected:
            raise ValueError(f"{path}: expected {expected} test rows in {group}")
        if train_ids & {x["set_id"] for x in meta["instances"]}:
            raise ValueError(f"{path}: training/evaluation overlap")
        if data["test_ids"].get(group) != [x["id"] for x in meta["instances"]]:
            raise ValueError(f"{path}: mismatched evaluation rows")
    by_cell = {}
    for row in data["results"]:
        by_cell.setdefault((row["position"], row["layer"]), []).append(row)
    for rows in by_cell.values():
        if len(rows) != 3 or {r["seed"] for r in rows} != {0, 1, 2}:
            raise ValueError(f"{path}: incomplete probe seeds")
        for row in rows:
            for group in data["test_ids"]:
                if not 0 <= row["eval"][group]["accuracy"] <= 1 or not 0 <= row["control_acc"].get(group, -1) <= 1:
                    raise ValueError(f"{path}: invalid accuracy or missing control")
            own = row["eval"][f"incongruent@{data['role']}"]
            if not math.isfinite(own["margin_mean"]):
                raise ValueError(f"{path}: non-finite probe readout")
    positions = set(train["pos_labels"]) if "alltok" in suffix else {"cotpre@v1", "cotpre@v2"}
    if set(by_cell) != {(position, layer) for position in positions for layer in train["layers"]}:
        raise ValueError(f"{path}: incomplete position/layer sweep")
    with zipfile.ZipFile(path.with_suffix(".npz")) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"{path}: corrupt per-instance outputs")
    own_group = f"incongruent@{data['role']}"
    with np.load(path.with_suffix(".npz")) as arrays:
        for position, layer in by_cell:
            for seed in (0, 1, 2):
                margin = arrays[f"{position}/L{layer}/s{seed}/{own_group}/margin"]
                if len(margin) != len(data["test_ids"][own_group]) or not np.isfinite(margin).all():
                    raise ValueError(f"{path}: incomplete or non-finite per-instance margins")
    return data, train


def promote_probe(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and json.loads(target.read_text())["train_dir"] != json.loads(source.read_text())["train_dir"]:
        raise ValueError(f"refusing to overwrite an unrelated probe: {target}")
    for extension in (".json", ".npz"):
        staged = target.with_suffix(extension + ".round4.tmp")
        shutil.copyfile(source.with_suffix(extension), staged)
        staged.replace(target.with_suffix(extension))


def value_step(data, model, role):
    cells = {}
    for row in data["results"]:
        if row["position"] == f"cotpre@{role}":
            cells.setdefault(row["layer"], []).append(row)
    layer = max(cells, key=lambda layer: mean(r["eval"]["neutral"]["accuracy"] -
                    r["control_acc"]["neutral"] for r in cells[layer]))
    rows = cells[layer]
    return {"model": model, "role": role, "layer": layer,
            "neutral_accuracy": mean(r["eval"]["neutral"]["accuracy"] for r in rows),
            "neutral_control": mean(r["control_acc"]["neutral"] for r in rows),
            "misleading_accuracy": mean(r["eval"][f"incongruent@{role}"]["accuracy"] for r in rows)}


def tick(item):
    path = ROOT / "CHECKLIST.md"
    text = re.sub(rf"(?m)^- \[ \] (?=(?:\*\*)?{item}[.:])", "- [x] ", path.read_text())
    path.write_text(text)


def run_stage(report, command, cwd=ROOT):
    log = LOG / f"collect_stage{len(report['stages'])+1}.out"
    with log.open("w") as output:
        result = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    report["stages"].append({"argv": command, "exit_code": result.returncode, "log": str(log)})
    if result.returncode:
        raise RuntimeError(f"{command}: exit {result.returncode}; see {log}")


def insert_extensions(report):
    rows = report["G2"]["cells"]
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Model & Variable & Layer & Neutral & Control & Misleading \\", r"\midrule"]
    for row in rows:
        role = "queried" if row["role"] == "v1" else "intermediate"
        lines.append(f"{MODEL[row['model']]} & {role} & {row['layer']} & " +
                     f"{100*row['neutral_accuracy']:.1f} & {100*row['neutral_control']:.1f} & " +
                     f"{100*row['misleading_accuracy']:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/probe_extensions.tex").write_text("\n".join(lines) + "\n")
    path = ROOT / "paper/main.tex"
    text = path.read_text()
    text = text.replace("including idle time within allocations.",
                        "including idle time and shared workers that also ran companion-code experiments.")
    marker = "% Round-four probe extensions"
    if marker not in text:
        section = marker + r"""
\section{Additional value readouts}
\label{app:probe-extensions}

We extend the value-step comparison to Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct and
Gemma-3-12B-Instruct. Each probe trains on 10,000 separate neutral problems with three probe
seeds, using demonstration set 7. Table~\ref{tab:probe-extensions} reports the added pairs;
the main comparison includes them. We select layers by neutral selectivity, as in the original
analysis. The misleading-name accuracy uses the same selected layer. This extension tests
the aggregate association; it does not resolve the within-problem link.

\begin{table*}[t]
\centering\small
\tabinput{probe_extensions}
\caption{Additional probes at each variable's value-writing position. Neutral and misleading
columns are true-digit accuracy; control is neutral name-identity accuracy. Rates are percentages,
averaged over three probe seeds. The layer maximizes neutral accuracy minus control accuracy.}
\label{tab:probe-extensions}
\end{table*}

We also probe every token for Gemma-3-4B-Instruct (Figures~\ref{fig:gemma-query}
and~\ref{fig:gemma-intermediate}), using 4,000 independent neutral training problems and
1,000 test problems per condition. The tokens shown come from one example; the curves and
heatmaps pool problems with different names and values. Maximum-over-layers curves describe
where a digit can be read; they do not identify the representation used to answer.
On neutral problems, the best pre-chain accuracy is \NUM{gemma-token-pre-v1}\% for the
queried digit and \NUM{gemma-token-pre-v2}\% for the intermediate digit. The queried readout
\NUM{gemma-token-timing-v1}; the intermediate readout \NUM{gemma-token-timing-v2}.
Token zero is the first token of the chain; negative positions are in the prompt.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_tokens_gemma3-4b-it_L3_cot_v1.pdf}
\caption{Gemma-3-4B-Instruct, queried variable, misleading-name problems under the correct
chain. Top: maximum-over-layers true-value accuracy and lure rate; bottom: true-value accuracy
by layer. Shading marks the variable's name; the dotted line marks the position before its
value; the arrow marks the supplied digit itself.}
\label{fig:gemma-query}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_tokens_gemma3-4b-it_L3_cot_v2.pdf}
\caption{The same Gemma token sweep for the intermediate variable. Both figures use
neutral-trained probes; no misleading-name problems enter training.}
\label{fig:gemma-intermediate}
\end{figure*}

"""
        anchor = r"\input{tables/figure_dump}"
        if anchor not in text:
            raise ValueError("diagnostic appendix anchor changed; cannot insert probe extensions")
        text = text.replace(anchor, section + anchor)
    path.write_text(text)


def update_association(report):
    """Weaken the interpretation if additional low-readout pairs lack a lure effect."""
    sweep = json.loads((RESULTS_DIR / "summary/seed_sweep_L3.json").read_text())
    cells = []
    for path in (RESULTS_DIR / "summary").glob("*/L3/cot/probes_grid.json"):
        model = path.parts[-4]
        grid = json.loads(path.read_text())
        for role in ("v1", "v2"):
            candidates = {key: value for key, value in grid.get(f"train_neutral/{role}", {}).items()
                          if key.startswith(f"cotpre@{role}|")}
            if not candidates:
                continue
            key = max(candidates, key=lambda key: candidates[key]["neutral"]["accuracy"][0] -
                      candidates[key]["neutral"]["control_acc"])
            contrast = sweep.get(f"{model}/cot", {}).get("contrasts", {}).get(f"lure_excess@{role}")
            if contrast:
                cells.append({"model": model, "role": role,
                              "neutral_accuracy": candidates[key]["neutral"]["accuracy"][0],
                              "lure_excess": contrast["pooled_mean"]})
    high = [row for row in cells if row["neutral_accuracy"] >= .9]
    low = [row for row in cells if row["neutral_accuracy"] < .9]
    report["value_step_association"] = {"cells": cells, "n_high": len(high), "n_low": len(low),
        "all_high_within_two_points": all(abs(row["lure_excess"]) <= .02 for row in high)}
    path = ROOT / "paper/main.tex"
    text = path.read_text()
    if len(low) > 1:
        text = text.replace("Reduced lure answers accompany accurate value readouts across the tested pairs. Yet the\nexception can state the correct value and still answer with the lure.",
                            "Accurate readouts accompany small lure effects; weak readouts do not consistently predict\ninterference. The exception can state the correct value and still answer with the lure.")
        text = text.replace(r"\subsection{The value readout distinguishes the exception}",
                            r"\subsection{Accurate readouts accompany low lure excess}")
        text = text.replace("This supports a\nlink between a decodable value and reduced lure answers.",
                            "Other low-readout pairs also lack a lure effect, so decodability alone does not\nseparate protected and unprotected models.")
        text = text.replace("the probed model--variable pairs, low lure excess accompanies an accurate value readout at\nthe step that writes it. This is an association; the readout need not drive the answer.",
                            "the probed pairs, accurate value readouts accompany low lure excess. Some pairs with\nweak readouts are also protected; decodability alone does not explain the pattern.")
        text = text.replace("Across the\nprobed pairs, this reduction accompanies an accurately decodable value at the writing step;",
                            "Accurate\nvalue readouts accompany small lure effects, but some weak-readout pairs are also protected;")
        text = text.replace("The value-readout association holds across the probed\npairs but is unresolved within instances and may reflect overall task competence.",
                            "The value-readout association does not fully separate protected and unprotected pairs;\nit remains unresolved within instances and may reflect overall task competence.")
    if not report["value_step_association"]["all_high_within_two_points"]:
        raise ValueError("new high-readout pair exceeds the two-point margin; the manuscript requires a claims review")
    path.write_text(text)


def main():
    report = {"collected_utc": datetime.now(timezone.utc).isoformat(), "workers": ["439652", "439653"],
              "status": "needs_attention", "stages": [], "errors": [], "G2": {"complete": False, "cells": []},
              "G3": {"complete": False}, "V1": {"complete": False}, "V2": {"complete": False}}
    destination = RESULTS_DIR / "summary/round4_completion.json"
    try:
        run_stage(report, [sys.executable, "scripts/72_disjoint_completion.py"], CODE)
        report["V1"]["complete"] = report["V2"]["complete"] = True
        tick("V1"); tick("V2")
    except Exception as error:
        report["errors"].append({"experiment": "V1/V2", "error": str(error)})
    try:
        sources = []
        for model in PANEL:
            for role in ("v1", "v2"):
                suffix = "__r5_fp32" if model == "gemma3-12b-it" else "__r4"
                source = OUT_DIR / "probes" / model / f"L3/cot/train_neutral{suffix}" / f"{role}.json"
                data, train = validate_probe(source, 10000, suffix)
                report["G2"]["cells"].append(value_step(data, model, role))
                sources.append((source, source.parent.parent / "train_neutral" / source.name))
        for source, target in sources:
            promote_probe(source, target)
        report["G2"]["complete"] = True
        run_stage(report, [sys.executable, "scripts/50_analysis.py", "--models", *PANEL, "--level", "3"])
        robust = load_module("round4_robust", ROOT / "scripts/58_robust_stats.py")
        path = RESULTS_DIR / "summary/robust_stats_L3.json"
        current = json.loads(path.read_text())
        current["link"] = robust.collect_links()
        path.write_text(json.dumps(current, indent=2) + "\n")
    except Exception as error:
        report["G2"]["complete"] = False
        report["errors"].append({"experiment": "G2", "error": str(error)})
    try:
        token = {}
        for role in ("v1", "v2"):
            source = OUT_DIR / "probes/gemma3-4b-it/L3/cot/train_neutral__alltok__r5_fp32" / f"{role}.json"
            data, train = validate_probe(source, 4000, "__alltok__r5_fp32")
            token[role] = {"source": str(source), "n_train": train["n"], "n_positions": len(train["pos_labels"]),
                           "n_neutral_test": len(data["test_ids"]["neutral"])}
            for script in ("53_kudo_figs.py", "58_token_figure.py"):
                run_stage(report, [sys.executable, f"scripts/{script}", "--model", "gemma3-4b-it",
                          "--level", "3", "--regime", "cot", "--role", role, "--cache-suffix", "__r5_fp32"])
        report["G3"] = {"complete": True, "roles": token}
    except Exception as error:
        report["errors"].append({"experiment": "G3", "error": str(error)})
    if report["G2"]["complete"] and report["G3"]["complete"]:
        try:
            insert_extensions(report)
            update_association(report)
            run_stage(report, [sys.executable, "scripts/70_compute_budget.py"])
            run_stage(report, [sys.executable, "scripts/51_tables.py"])
            run_stage(report, [sys.executable, "scripts/99_selfcheck.py"])
            run_stage(report, ["make", "paper-submission"])
            tick("G2"); tick("G3")
        except Exception as error:
            report["errors"].append({"experiment": "arithmetic_rebuild", "error": str(error)})
    report["status"] = "complete" if not report["errors"] else "needs_attention"
    if report["status"] == "complete":
        path = ROOT / "CHECKLIST.md"
        text = path.read_text().replace("- [ ] Collect validated outputs, update claims to match them, regenerate figures and build both papers.",
                                       "- [x] Collect validated outputs, update claims to match them, regenerate figures and build both papers.")
        path.write_text(text)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    with (RESULTS_DIR / "NOTES.md").open("a") as output:
        output.write("\n### Round-four collection\n\n" + f"Status: {report['status']}. " +
                     "Audited cell scores, stages and failures are recorded in `summary/round4_completion.json`. " +
                     "Only validated, rebuilt experiments are ticked in the checklist.\n")
    print(json.dumps({"status": report["status"], "errors": report["errors"]}, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
