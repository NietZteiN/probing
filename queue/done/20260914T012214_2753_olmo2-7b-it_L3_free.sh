#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model olmo2-7b-it --level 3 --regimes free --no-train --no-forced --groups letter neutral congruent@v1 congruent@v2 incongruent@v1 incongruent@v2

# worker=394304 rc=0 finished=2026-09-14T04:08:13Z
