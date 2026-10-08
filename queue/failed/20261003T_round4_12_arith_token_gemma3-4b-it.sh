#!/bin/bash
set -euo pipefail
cd /work/jvl210002/migration/probing
source scripts/env.sh
python scripts/20_run_model.py --model gemma3-4b-it --level 3 --regimes cot --groups train_neutral neutral incongruent@v1 incongruent@v2 --cache-only --all-positions --cache-suffix __r4 --train-limit 4000 --limit 1000 --batch-size 4 > log/round4_2026-10-02/arith_token_gemma3-4b-it_stage1.out 2>&1
python scripts/30_train_probes.py --model gemma3-4b-it --level 3 --regime cot --roles v1 v2 --suffix __alltok__r4 --test-groups neutral incongruent@v1 incongruent@v2 > log/round4_2026-10-02/arith_token_gemma3-4b-it_stage2.out 2>&1

# worker=439652 rc=143 finished=2026-10-03T06:42:27Z
