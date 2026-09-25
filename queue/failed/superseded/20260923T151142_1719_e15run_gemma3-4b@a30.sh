#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model gemma3-4b --level 3 --regimes cot direct

# worker=421483 rc=1 finished=2026-09-23T23:12:04Z
