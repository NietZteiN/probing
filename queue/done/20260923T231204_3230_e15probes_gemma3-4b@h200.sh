#!/bin/bash
# needs: /scratch/juno/jvl210002/probing/runs/gemma3-4b/L3/cot/incongruent@v2/summary.json
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/30_train_probes.py --model gemma3-4b --level 3 --regime cot --train train_neutral

# worker=422443 rc=0 finished=2026-09-24T19:21:16Z
