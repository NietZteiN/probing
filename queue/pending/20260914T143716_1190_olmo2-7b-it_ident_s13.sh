#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model olmo2-7b-it --level 3 --data-suffix _ident --regimes cot_s13 direct_s13 --no-train --no-forced
