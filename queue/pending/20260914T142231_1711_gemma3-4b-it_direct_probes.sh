#!/bin/bash
set -uo pipefail
cd "/work/jvl210002/migration/probing"
python scripts/30_train_probes.py --model gemma3-4b-it --level 3 --regime direct --train train_neutral
