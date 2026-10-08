#!/bin/bash
# needs: /scratch/juno/jvl210002/codecue/round6_identifiers/screen/gemma3-4b-it.json
# needs: /scratch/juno/jvl210002/codecue/round6_identifiers/screen/codegemma-7b-it.json
# needs: /scratch/juno/jvl210002/codecue/round6_identifiers/screen/llama31-8b-it.json
# needs: /scratch/juno/jvl210002/codecue/round6_identifiers/screen/olmo2-1b-it.json
# needs: /scratch/juno/jvl210002/codecue/round6_identifiers/screen/olmo2-7b-it.json
# needs: /scratch/juno/jvl210002/codecue/round6_identifiers/screen/codellama-7b-it.json
# needs: /scratch/juno/jvl210002/codecue/round6_identifiers/screen/llama32-3b-it.json
set -euo pipefail
cd /work/jvl210002/migration/probing
python ../codecue/scripts/79_screened_replication.py --stage freeze

# worker=440673 rc=0 finished=2026-10-03T23:12:33Z
