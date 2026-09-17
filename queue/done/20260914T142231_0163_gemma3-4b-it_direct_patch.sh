#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model gemma3-4b-it --level 3 --regime direct --limit 500

# worker=397743 rc=0 finished=2026-09-14T17:58:03Z
