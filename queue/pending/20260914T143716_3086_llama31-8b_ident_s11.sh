#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model llama31-8b --level 3 --data-suffix _ident --regimes cot_s11 direct_s11 --no-train --no-forced
