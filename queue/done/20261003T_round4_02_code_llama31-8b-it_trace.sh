#!/bin/bash
set -euo pipefail
cd /work/jvl210002/migration/codecue
source scripts/env.sh
python scripts/30_cache_and_probe.py --model llama31-8b-it --level 5 --role v1 --regime trace --n-train 2000 --cell-only --positions pre@v1 --batch-size 8 > log/round4_2026-10-02/code_llama31-8b-it_trace_stage1.out 2>&1

# worker=439652 rc=0 finished=2026-10-03T05:13:06Z
