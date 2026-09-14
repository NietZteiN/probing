#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model llama31-8b-it --level 3 --regimes cot direct --no-train --no-forced
