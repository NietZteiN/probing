#!/usr/bin/env python
"""The whole campaign as dependent SLURM jobs, per model.

    python scripts/slurm/pipeline.py --stage all --models llama32-3b llama31-8b [--levels 3] [--dry-run]
    python scripts/slurm/pipeline.py --stage tokcheck --models all
    python scripts/slurm/pipeline.py --stage run|probes|patch --models llama32-3b [--run-dep <jobid>]

Stages and dependencies:
    tokcheck   dev partition, CPU: rule-4 word lists (must precede `make data`)
    run        one GPU job per (model, level): behaviour + caches, both regimes
    probes     one GPU job per (model, level, regime, train split), afterok:run
    patch      one GPU job per (model, level, regime), afterok:run
Partition per model from configs/models.yaml; a30 and h100 carry no QoS, so these never touch
the 4-job `juno` pool the account's other projects share on h200.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "slurm"))
from cueconf.config import load_config  # noqa: E402
from submit import submit  # noqa: E402

PY = "python"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "tokcheck", "run", "probes", "patch"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[3])
    ap.add_argument("--regimes", nargs="+", default=["cot", "direct"])
    ap.add_argument("--probe-train", nargs="+", default=["train_neutral", "train_letter"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-dep", help="existing run job id to hang probes/patch on")
    ap.add_argument("--after", help="job id the RUN stage must wait for (afterok), e.g. a smoke test")
    ap.add_argument("--partition", help="override the per-model partition from models.yaml (e.g. h200 when a30/h100 are full)")
    ap.add_argument("--time", help="override the partition's default walltime, e.g. 04:00:00")
    a = ap.parse_args()
    models = load_config("models.yaml")["models"]
    keys = list(models) if a.models == ["all"] else a.models
    for k in keys:
        if k not in models:
            raise SystemExit(f"unknown model {k}")
    if a.stage == "tokcheck":
        submit("tokcheck", "dev", [PY, "scripts/11_tokenizer_check.py"] + [x for k in keys for x in ("--model", k)], dry_run=a.dry_run)
        return 0
    for k in keys:
        part = a.partition or models[k]["partition"]
        for level in a.levels:
            run_id = a.run_dep
            if a.stage in ("all", "run"):
                run_id = submit(f"run_{k}_L{level}", part,
                                [PY, "scripts/20_run_model.py", "--model", k, "--level", str(level), "--regimes", *a.regimes],
                                dependency=(f"afterok:{a.after}" if a.after else None), time=a.time,
                                dry_run=a.dry_run) or run_id
            dep = f"afterok:{run_id}" if run_id else None
            for regime in a.regimes:
                if a.stage in ("all", "probes"):
                    for tr in a.probe_train:
                        submit(f"probes_{k}_L{level}_{regime}_{tr}", part,
                               [PY, "scripts/30_train_probes.py", "--model", k, "--level", str(level), "--regime", regime, "--train", tr],
                               dependency=dep, time=a.time, dry_run=a.dry_run)
                if a.stage in ("all", "patch"):
                    submit(f"patch_{k}_L{level}_{regime}", part,
                           [PY, "scripts/40_patch.py", "--model", k, "--level", str(level), "--regime", regime],
                           dependency=dep, time=a.time, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
