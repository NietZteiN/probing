#!/bin/bash
# needs: /scratch/juno/jvl210002/probing/runs/olmo2-1b-it/L3/direct/train_neutral
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model olmo2-1b-it --level 3 --regime direct --contrasts main ctl_word ctl_lure inject --limit 500

# worker=397744 rc=0 finished=2026-09-14T23:54:17Z
