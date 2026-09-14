#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model gemma3-4b-it --level 3 --regimes cot direct --no-train --no-forced

# worker=394303 rc=0 finished=2026-09-14T02:02:23Z
