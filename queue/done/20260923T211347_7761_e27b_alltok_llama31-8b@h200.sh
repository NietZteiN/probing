#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model llama31-8b --level 3 --all-positions --layer-stride 2 --train-limit 4000 --groups train_neutral train_letter letter neutral congruent@v1 congruent@v2 incongruent@v1 incongruent@v2

# worker=421482 rc=0 finished=2026-09-24T10:11:11Z
