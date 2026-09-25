#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model llama31-8b --level 3 --regimes cot --all-positions --layer-stride 2 --train-limit 4000 --groups letter neutral congruent@v1 congruent@v2 incongruent@v1 incongruent@v2

# worker=420833 rc=0 finished=2026-09-23T16:18:07Z
