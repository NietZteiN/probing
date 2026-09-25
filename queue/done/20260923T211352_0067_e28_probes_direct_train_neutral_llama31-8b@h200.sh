#!/bin/bash
# needs: /scratch/juno/jvl210002/probing/runs/llama31-8b/L3/direct/train_neutral__alltok/summary.json
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/30_train_probes.py --model llama31-8b --level 3 --regime direct --train train_neutral --suffix __alltok

# worker=422443 rc=0 finished=2026-09-24T16:54:27Z
