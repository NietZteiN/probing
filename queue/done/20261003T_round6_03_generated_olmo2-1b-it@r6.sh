#!/bin/bash
# needs: /work/jvl210002/migration/probing/log/round6_2026-10-03/arithmetic_ready.ok
# needs: /work/jvl210002/migration/probing/log/round6_2026-10-03/precision_ready.ok
set -euo pipefail
cd /work/jvl210002/migration/probing
python scripts/80_generated_trained_probes.py --model olmo2-1b-it --smoke
python scripts/80_generated_trained_probes.py --model olmo2-1b-it

# worker=440673 rc=0 finished=2026-10-03T23:48:16Z
