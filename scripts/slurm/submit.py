#!/usr/bin/env python
"""Submit one command to SLURM with the project env. Small on purpose: SLURM already does
allocation, isolation and accounting; what this adds is the env, the log path, a manifest in the
log directory (argv, git sha, submit time, job id, node) and a trap that stamps the exit code
into it, so a walltime kill is visible rather than a job silently missing from done/.

    python scripts/slurm/submit.py --name tok --partition dev -- python scripts/11_tokenizer_check.py --model llama32-3b
    python scripts/slurm/submit.py --name run --partition a30 --dependency afterok:123 -- python scripts/20_run_model.py ...
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cueconf.config import load_config  # noqa: E402

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
{gres}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
{exclude}{dependency}#SBATCH --output={log_dir}/%j_{name}.out
#SBATCH --error={log_dir}/%j_{name}.out
set -uo pipefail
source {root}/scripts/env.sh
cd {root}
MANIFEST="{log_dir}/${{SLURM_JOB_ID}}_{name}.json"
python {root}/scripts/slurm/stamp.py start "{manifest_src}" "$MANIFEST"
finish() {{ rc=$?; python {root}/scripts/slurm/stamp.py finish "$MANIFEST" "$rc"; exit $rc; }}
# a walltime kill arrives as SIGTERM; record it as 143 (the EXIT trap alone logged 0 for job 391916)
trap 'python {root}/scripts/slurm/stamp.py finish "$MANIFEST" 143; exit 143' TERM
trap finish EXIT
echo "# probing job $SLURM_JOB_ID on $SLURMD_NODENAME ($SLURM_JOB_PARTITION) $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "# no GPU"
echo "# {argv}"
{argv}
"""


def submit(name: str, partition: str, argv: list[str], dependency: str | None = None, time: str | None = None,
           mem: str | None = None, dry_run: bool = False) -> str | None:
    cfg = load_config("compute.yaml")["slurm"]
    p = cfg["partitions"][partition]
    log_dir = ROOT / cfg["log_dir_rel"]
    log_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_src = log_dir / f"submit_{stamp}_{name}.json"
    argv_s = " ".join(shlex.quote(x) for x in argv)
    script = TEMPLATE.format(
        name=name, partition=partition, gres=(f"#SBATCH --gres={p['gres']}\n" if p.get("gres") else ""),
        cpus=p["cpus"], mem=mem or p["mem"], time=time or p["time"],
        exclude=(f"#SBATCH --exclude={p['exclude']}\n" if p.get("exclude") else ""),
        dependency=(f"#SBATCH --dependency={dependency}\n" if dependency else ""),
        log_dir=log_dir, root=ROOT, manifest_src=manifest_src, argv=argv_s)
    if dry_run:
        print(f"--- would submit {name} on {partition} (dep={dependency}):\n    {argv_s}")
        return None
    manifest_src.write_text(json.dumps({"name": name, "partition": partition, "argv": argv, "git": sha,
                                        "submitted_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    sfile = log_dir / f"submit_{stamp}_{name}.sbatch"
    sfile.write_text(script)
    out = subprocess.run(["sbatch", "--parsable", str(sfile)], capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr); raise SystemExit(out.returncode)
    jid = out.stdout.strip().split(";")[0]
    print(f"submitted {name} -> job {jid} ({partition})")
    return jid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--dependency")
    ap.add_argument("--time")
    ap.add_argument("--mem")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("argv", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    argv = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
    submit(a.name, a.partition, argv, a.dependency, a.time, a.mem, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
