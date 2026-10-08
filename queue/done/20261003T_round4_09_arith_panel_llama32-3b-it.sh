#!/bin/bash
set -euo pipefail
cd /work/jvl210002/migration/probing
source scripts/env.sh
python scripts/20_run_model.py --model llama32-3b-it --level 3 --regimes cot --groups train_neutral neutral incongruent@v1 incongruent@v2 --cache-only --cache-suffix __r4 --batch-size 4 > log/round4_2026-10-02/arith_panel_llama32-3b-it_stage1.out 2>&1
python scripts/30_train_probes.py --model llama32-3b-it --level 3 --regime cot --roles v1 v2 --suffix __r4 --positions cotpre@v1 cotpre@v2 --test-groups neutral incongruent@v1 incongruent@v2 > log/round4_2026-10-02/arith_panel_llama32-3b-it_stage2.out 2>&1

# worker=439653 rc=0 finished=2026-10-03T05:34:26Z
