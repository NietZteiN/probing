#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model llama31-8b --level 3 --regime cot --contrasts inject --limit 500

# worker=394304 rc=0 finished=2026-09-14T04:26:57Z
