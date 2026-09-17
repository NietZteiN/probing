#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model llama31-8b --level 3 --data-suffix _ident --regimes cot_s11 direct_s11 --no-train --no-forced


# worker=397744 rc=0 finished=2026-09-14T22:33:31Z
