#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model olmo2-7b-it --level 3 --data-suffix _ident --regimes cot_s11 direct_s11 --no-train --no-forced


# worker=397743 rc=0 finished=2026-09-14T22:38:30Z
