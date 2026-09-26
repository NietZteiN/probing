#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/20_run_model.py --model nemotron-nano-8b --level 3 --regimes direct cot free --groups neutral congruent@v2 incongruent@v2 --no-train --no-forced

# worker=425301 rc=0 finished=2026-09-26T14:56:13Z
