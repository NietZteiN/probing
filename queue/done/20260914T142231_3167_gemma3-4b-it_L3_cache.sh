#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model gemma3-4b-it --level 3 --regimes cot direct --groups train_neutral letter neutral congruent@v1 congruent@v2 incongruent@v1 incongruent@v2 --overwrite

# worker=397743 rc=0 finished=2026-09-14T18:15:35Z
