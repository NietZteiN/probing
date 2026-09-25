#!/bin/bash
# needs: /scratch/juno/jvl210002/probing/runs/llama31-8b/L3/cot/train_neutral__alltok/summary.json
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/30_train_probes.py --model llama31-8b --level 3 --regime cot --train train_neutral --suffix __alltok

# worker=421482 rc=0 finished=2026-09-24T13:56:12Z
