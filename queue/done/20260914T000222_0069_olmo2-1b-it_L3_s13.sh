#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model olmo2-1b-it --level 3 --regimes cot_s13 direct_s13 --no-train --no-forced

# worker=394303 rc=0 finished=2026-09-14T00:57:16Z
