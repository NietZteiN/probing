#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model llama31-8b --level 3 --data-suffix _ident --regimes cot_s13 direct_s13 --no-train --no-forced

# worker=397743 rc=0 finished=2026-09-14T23:10:26Z
