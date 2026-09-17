#!/bin/bash
# needs: /scratch/juno/jvl210002/probing/runs/olmo2-1b-it/L3/cot/train_neutral
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model olmo2-1b-it --level 3 --regime cot --contrasts inject ctl_word --limit 500

# worker=397744 rc=0 finished=2026-09-15T00:01:19Z
