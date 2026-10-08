#!/bin/bash
set -euo pipefail
cd /work/jvl210002/migration/probing
source scripts/env.sh
python scripts/20_run_model.py --model gemma3-12b-it --level 3 --regimes cot --groups train_neutral neutral incongruent@v1 incongruent@v2 --cache-only --cache-suffix __r5_fp32 --batch-size 4 > log/round5_2026-10-03/gemma12_fp32_cache.out 2>&1
python scripts/30_train_probes.py --model gemma3-12b-it --level 3 --regime cot --roles v1 v2 --suffix __r5_fp32 --positions cotpre@v1 cotpre@v2 --test-groups neutral incongruent@v1 incongruent@v2 > log/round5_2026-10-03/gemma12_fp32_probes.out 2>&1
python scripts/76_competence_followup.py > log/round5_2026-10-03/competence.out 2>&1

# worker=439652 rc=0 finished=2026-10-03T07:33:54Z
