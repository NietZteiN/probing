#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model gemma3-12b-it --level 3 --regimes cot_s11 direct_s11 --no-train --no-forced

# worker=394304 rc=0 finished=2026-09-14T01:53:30Z
