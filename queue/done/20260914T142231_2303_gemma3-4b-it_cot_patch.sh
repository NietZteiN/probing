#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model gemma3-4b-it --level 3 --regime cot --contrasts inject ctl_word --limit 500

# worker=397744 rc=0 finished=2026-09-14T18:16:57Z
