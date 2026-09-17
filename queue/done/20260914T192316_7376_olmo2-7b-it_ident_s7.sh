#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model olmo2-7b-it --level 3 --data-suffix _ident --regimes cot direct --no-train --no-forced

# worker=397744 rc=0 finished=2026-09-14T23:17:30Z
