#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model gemma3-4b --level 3 --regimes cot_s11 direct_s11 --no-train --no-forced

# worker=424658 rc=0 finished=2026-09-25T20:52:31Z
