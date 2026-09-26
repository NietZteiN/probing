#!/usr/bin/env python
"""GPU hours this project used, from SLURM accounting, for the paper's compute statement (ARR C1).
    python scripts/70_compute_budget.py
Every job this repo submits leaves log/slurm/<jobid>_<name>.json; the job ids come from there, so
jobs of other projects on the same account are never counted. Elapsed time is what SLURM
allocated, so a queue worker's idle minutes are included: the figure is an upper bound on the
compute the experiments needed. Writes results/summary/compute.json.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402


def main() -> int:
    ids = sorted({m.group(1) for p in (PROJECT_ROOT / "log" / "slurm").glob("*.json")
                  if (m := re.match(r"(\d+)_", p.name))})
    if not ids:
        print("no job manifests under log/slurm"); return 1
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "-o", "JobID,Partition,ElapsedRaw,AllocTRES"],
                         capture_output=True, text=True, check=True).stdout
    hours, jobs = Counter(), Counter()
    for line in out.splitlines():
        _, part, elapsed, tres = line.split("|")
        m = re.search(r"gres/gpu=(\d+)", tres)
        if m:
            hours[part] += int(m.group(1)) * int(elapsed or 0) / 3600
            jobs[part] += 1
    rec = {"gpu_hours": round(sum(hours.values()), 1),
           "by_partition": {p: {"gpu_hours": round(h, 1), "jobs": jobs[p]} for p, h in sorted(hours.items())},
           "manifests": len(ids), "note": "allocated time incl. idle queue workers; an upper bound"}
    (RESULTS_DIR / "summary" / "compute.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
