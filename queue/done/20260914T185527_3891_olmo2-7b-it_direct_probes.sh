#!/bin/bash
# needs: /scratch/juno/jvl210002/probing/runs/olmo2-7b-it/L3/direct/train_neutral
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/30_train_probes.py --model olmo2-7b-it --level 3 --regime direct --train train_neutral


# worker=397743 rc=0 finished=2026-09-14T21:55:11Z
