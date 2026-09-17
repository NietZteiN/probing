#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model olmo2-7b-it --level 3 --regime direct --limit 500

# worker=397744 rc=0 finished=2026-09-14T17:46:32Z
