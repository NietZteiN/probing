#!/bin/bash
# needs: /work/jvl210002/migration/probing/log/round6_2026-10-03/arithmetic_ready.ok
# needs: /work/jvl210002/migration/probing/log/round6_2026-10-03/precision_ready.ok
set -euo pipefail
cd /work/jvl210002/migration/probing
python scripts/80_generated_trained_probes.py --model llama32-3b --smoke
python scripts/80_generated_trained_probes.py --model llama32-3b

# worker=440673 rc=0 finished=2026-10-03T23:44:26Z
