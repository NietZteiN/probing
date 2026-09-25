#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model llama31-8b --level 3 --regime direct --grid --sources other neutral incongruent_alt

# worker=420833 rc=0 finished=2026-09-23T17:39:14Z
