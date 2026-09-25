#!/bin/bash
# needs: /scratch/juno/jvl210002/probing/runs/gemma3-4b/L3/cot/incongruent@v2/summary.json
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model gemma3-4b --level 3 --regime cot

# worker=421482 rc=0 finished=2026-09-24T20:30:07Z
