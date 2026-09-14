#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model llama31-8b --level 3 --regime direct --contrasts inject --limit 500
