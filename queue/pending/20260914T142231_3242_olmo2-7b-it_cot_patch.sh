#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/40_patch.py --model olmo2-7b-it --level 3 --regime cot --contrasts inject ctl_word --limit 500
