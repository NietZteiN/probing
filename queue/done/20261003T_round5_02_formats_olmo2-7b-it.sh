#!/bin/bash
# needs: /work/jvl210002/migration/probing/log/round5_2026-10-03/gpu_preflight.ok
set -euo pipefail
cd /work/jvl210002/migration/codecue
source scripts/env.sh
python scripts/75_format_controls.py --model olmo2-7b-it > log/round5_2026-10-03/formats_olmo2-7b-it.out 2>&1

# worker=439653 rc=0 finished=2026-10-03T06:40:39Z
