#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model gemma3-4b-it --level 3 --data-suffix _ident --regimes cot direct --no-train --no-forced
