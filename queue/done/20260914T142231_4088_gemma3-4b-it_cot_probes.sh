#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/30_train_probes.py --model gemma3-4b-it --level 3 --regime cot --train train_neutral

# worker=397744 rc=0 finished=2026-09-14T19:53:50Z
