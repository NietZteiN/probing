#!/bin/bash
# needs: /scratch/juno/jvl210002/codecue/round6_identifiers/frozen_names.json
# needs: /work/jvl210002/migration/probing/log/round6_2026-10-03/code_ready.ok
set -euo pipefail
cd /work/jvl210002/migration/probing
python ../codecue/scripts/79_screened_replication.py --stage run --model codegemma-7b-it

# worker=440673 rc=0 finished=2026-10-03T23:19:11Z
