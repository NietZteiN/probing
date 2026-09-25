#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model gemma3-4b --level 3 --regimes cot direct

# worker=421483 rc=0 finished=2026-09-24T01:51:18Z
