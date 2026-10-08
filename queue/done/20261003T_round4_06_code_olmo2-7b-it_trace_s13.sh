#!/bin/bash
set -euo pipefail
cd /work/jvl210002/migration/codecue
source scripts/env.sh
python scripts/30_cache_and_probe.py --model olmo2-7b-it --level 5 --role v1 --regime trace_s13 --n-train 2000 --cell-only --positions pre@v1 --batch-size 8 > log/round4_2026-10-02/code_olmo2-7b-it_trace_s13_stage1.out 2>&1

# worker=439653 rc=0 finished=2026-10-03T05:16:44Z
