#!/usr/bin/env python
"""Collect the requested round-3 GPU controls, publish their aggregates, and update status.

Submit on the CPU dev partition with afterany:<GPU job>. Missing or incomplete GPU outputs
leave G1/C5 unchecked and produce an explicit failure report rather than a success tick.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cueconf.config import OUT_DIR, RESULTS_DIR  # noqa: E402


def tick(item: str) -> None:
    path = ROOT / "CHECKLIST.md"
    source = path.read_text()
    source = re.sub(rf"(?m)^- \[ \] (?=(?:\*\*)?{item}[.:])", "- [x] ", source)
    path.write_text(source)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-job", required=True)
    a = ap.parse_args()
    report_path = RESULTS_DIR / "summary" / "round3_completion.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["gpu_job"] = a.gpu_job
    report["collected_utc"] = datetime.now(timezone.utc).isoformat()
    stages = []
    try:
        patch_dir = OUT_DIR / "patching/olmo2-1b-it/L3/cot"
        controls = {}
        for contrast in ("main", "ctl_lure"):
            for target in ("v1", "v2"):
                key = f"{contrast}@{target}"
                path = patch_dir / f"{key}.json"
                raw = json.loads(path.read_text())
                if len(raw["rows"]) != 500 or raw["level"] != 3 or raw["regime"] != "cot":
                    raise ValueError(f"{key}: expected 500 pairs, level 3, CoT")
                controls[key] = raw["summary"]["ALL"]
        report["G1"] = {"complete": True, "all_layers": controls,
                        "readout": "teacher-forced gold chain; not the model's own generated chain"}
        commands = [
            [sys.executable, "scripts/50_analysis.py", "--models", "olmo2-1b-it", "--level", "3"],
            [sys.executable, "scripts/51_tables.py"],
            [sys.executable, "scripts/60_master_figure.py"],
            [sys.executable, "scripts/66_figure_dump.py"],
            [sys.executable, "scripts/99_selfcheck.py"],
            ["make", "paper"],
        ]
        for command in commands:
            log = ROOT / "log" / ("round3_final_" + str(len(stages) + 1) + ".out")
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            stages.append({"argv": command, "exit_code": result.returncode, "log": str(log.relative_to(ROOT))})
            if result.returncode:
                raise RuntimeError(f"{command}: exit {result.returncode}; see {log}")
        report["C5"] = {"complete": True, "stages": stages}
        report["status"] = "complete"
        tick("G1")
        tick("C5")
        source = (ROOT / "CHECKLIST.md").read_text()
        source = source.replace("**GPU status:** queued,", "**GPU status:** completed,")
        source = source.replace("Final G1/C5 ticks wait for its outputs and the dependent collection/build job.",
                                "G1/C5 outputs collected; final self-check and paper build passed.")
        (ROOT / "CHECKLIST.md").write_text(source)
    except Exception as error:
        report["status"] = "needs_attention"
        report["error"] = str(error)
        report["C5"] = {"complete": False, "stages": stages}
        if "G1" not in report or not report["G1"].get("complete"):
            report["G1"] = {"complete": False}
        print(f"Round-3 collection failed: {error}", file=sys.stderr)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "gpu_job", "error")}, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
