#!/bin/bash
# needs: /scratch/juno/jvl210002/probing/runs/olmo2-7b-it/L3/cot/train_neutral
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/30_train_probes.py --model olmo2-7b-it --level 3 --regime cot --train train_neutral


# worker=397744 rc=0 finished=2026-09-14T22:19:13Z
