#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model llama32-3b --level 3 --regime cot --contrasts inject --limit 500
