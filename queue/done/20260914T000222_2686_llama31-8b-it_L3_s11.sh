#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model llama31-8b-it --level 3 --regimes cot_s11 direct_s11 --no-train --no-forced

# worker=394303 rc=0 finished=2026-09-14T02:15:10Z
