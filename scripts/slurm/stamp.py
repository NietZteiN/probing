#!/usr/bin/env python
"""Job manifest stamping, called from the sbatch template: `start <src> <dst>` copies the submit
manifest into the job's own file with job id and node; `finish <dst> <rc>` adds the exit code."""
import datetime
import json
import os
import sys


def main() -> int:
    mode = sys.argv[1]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if mode == "start":
        src, dst = sys.argv[2], sys.argv[3]
        d = json.load(open(src))
        d.update(job_id=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME"), started_utc=now)
        json.dump(d, open(dst, "w"), indent=1)
    else:
        dst, rc = sys.argv[2], int(sys.argv[3])
        d = json.load(open(dst))
        d.update(exit_code=rc, finished_utc=now)
        json.dump(d, open(dst, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
