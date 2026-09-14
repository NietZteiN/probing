#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model llama32-3b-it --level 3 --regimes cot_s11 direct_s11 --no-train --no-forced
