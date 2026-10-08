#!/bin/bash
#SBATCH --job-name=r6_worker2
#SBATCH --partition=h100,h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=g-06-01
#SBATCH --output=/work/jvl210002/migration/probing/log/slurm/%j_r6_worker2.out
#SBATCH --error=/work/jvl210002/migration/probing/log/slurm/%j_r6_worker2.out
set -uo pipefail
source /work/jvl210002/migration/probing/scripts/env.sh
cd /work/jvl210002/migration/probing
MANIFEST="/work/jvl210002/migration/probing/log/slurm/${SLURM_JOB_ID}_r6_worker2.json"
python /work/jvl210002/migration/probing/scripts/slurm/stamp.py start "/work/jvl210002/migration/probing/log/round6_2026-10-03/r6_worker2_submission.json" "$MANIFEST"
finish() { rc=$?; python /work/jvl210002/migration/probing/scripts/slurm/stamp.py finish "$MANIFEST" "$rc"; exit $rc; }
# a walltime kill arrives as SIGTERM; record it as 143 (the EXIT trap alone logged 0 for job 391916)
trap 'python /work/jvl210002/migration/probing/scripts/slurm/stamp.py finish "$MANIFEST" 143; exit 143' TERM
trap finish EXIT
echo "# probing job $SLURM_JOB_ID on $SLURMD_NODENAME ($SLURM_JOB_PARTITION) $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "# no GPU"
echo "# env WORKER_IDLE_EXIT_MIN=10 bash scripts/slurm/worker.sh r6"
env WORKER_IDLE_EXIT_MIN=10 bash scripts/slurm/worker.sh r6
