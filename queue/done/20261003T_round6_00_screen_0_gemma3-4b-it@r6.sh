#!/bin/bash
# needs: /work/jvl210002/migration/probing/log/round6_2026-10-03/code_ready.ok
set -euo pipefail
cd /work/jvl210002/migration/probing
python ../codecue/scripts/79_screened_replication.py --stage screen --model gemma3-4b-it

# worker=440673 rc=0 finished=2026-10-03T23:08:16Z
