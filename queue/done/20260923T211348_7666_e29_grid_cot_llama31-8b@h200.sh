#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model llama31-8b --level 3 --regime cot --grid --sources other neutral incongruent_alt --limit 500

# worker=421482 rc=0 finished=2026-09-24T11:06:24Z
