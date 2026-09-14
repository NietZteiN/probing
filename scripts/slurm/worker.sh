#!/bin/bash
# A reserved GPU: one batch job that holds a card and runs queued tasks one at a time.
#
# WHY THIS AND NOT salloc. `salloc --no-shell` + `srun --jobid` fails from juno's login node
# ("Zero Bytes were transmitted or received", job 391543) while srun INSIDE a batch job works.
# So a reservation here is a long-lived batch job with a work loop, not an interactive alloc.
#
#   python scripts/slurm/submit.py --name worker1 --partition h200 --time 12:00:00 -- bash scripts/slurm/worker.sh
#   scripts/slurm/enqueue.sh "python scripts/20_run_model.py --model gemma3-4b-it ..."
# Stop a worker early by creating queue/STOP; tasks already claimed finish first.
set -uo pipefail
Q="${PROBE_ROOT:?}/queue"
mkdir -p "$Q"/{pending,running,done,failed}
ME="${SLURM_JOB_ID:-$$}"
echo "# worker $ME on $(hostname), queue $Q"
idle=0
while true; do
  [[ -f "$Q/STOP" ]] && { echo "# STOP file present, exiting"; break; }
  task=$(ls -1 "$Q/pending" 2>/dev/null | head -1)
  if [[ -z "$task" ]]; then
    idle=$((idle+1)); (( idle % 30 == 1 )) && echo "# idle, waiting for work ($(date -u +%H:%M:%S))"
    sleep 20; continue
  fi
  idle=0
  # claim atomically: mv fails if another worker got there first
  if ! mv "$Q/pending/$task" "$Q/running/$task" 2>/dev/null; then continue; fi
  echo "=== [$ME] $(date -u +%FT%TZ) START $task"; cat "$Q/running/$task"
  bash "$Q/running/$task" 2>&1; rc=$?
  echo "=== [$ME] $(date -u +%FT%TZ) END $task rc=$rc"
  dest=done; [[ $rc -ne 0 ]] && dest=failed
  printf '\n# worker=%s rc=%s finished=%s\n' "$ME" "$rc" "$(date -u +%FT%TZ)" >> "$Q/running/$task"
  mv "$Q/running/$task" "$Q/$dest/$task"
done
