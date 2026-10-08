#!/bin/bash
# needs: /work/jvl210002/migration/probing/log/round5_2026-10-03/gpu_preflight.ok
set -euo pipefail
cd /work/jvl210002/migration/probing
source scripts/env.sh
python scripts/75_generated_chain_followup.py --model llama32-3b > log/round5_2026-10-03/own_llama32-3b.out 2>&1

# worker=439653 rc=1 finished=2026-10-03T06:33:43Z
