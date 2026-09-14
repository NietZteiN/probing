#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model olmo2-7b-it --level 3 --regimes free --no-train --no-forced --groups letter neutral congruent@v1 congruent@v2 incongruent@v1 incongruent@v2
