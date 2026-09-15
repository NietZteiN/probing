#!/bin/bash
# A reserved GPU: one batch job that holds a card and runs queued tasks one at a time.
#
# NOTE: a running worker keeps executing the copy of this loop that bash parsed when it
# started. Editing this file does NOT change workers that are already running (2026-09-14: the
# "# needs:" defer below was added while two workers were up, and they ignored it, so a probe
# job ran before its cache job and failed). After changing this file, drain and resubmit the
# workers before relying on the new behaviour.
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
  # a task may declare `# needs: <path>` lines; defer it (to the back of the queue) until every
  # path exists, so probes never start before the cache job that feeds them
  missing=""
  while read -r need; do [[ -n "$need" && ! -e "$need" ]] && missing="$need"; done < <(sed -n 's/^# needs: *//p' "$Q/pending/$task" 2>/dev/null)
  if [[ -n "$missing" ]]; then
    later="$(date -u +%Y%m%dT%H%M%S)_${task#*_}"
    mv "$Q/pending/$task" "$Q/pending/$later" 2>/dev/null && echo "# deferred $task (needs $missing)"
    sleep 60; continue
  fi
  # claim atomically: mv fails if another worker got there first
  if ! mv "$Q/pending/$task" "$Q/running/$task" 2>/dev/null; then continue; fi
  echo "=== [$ME] $(date -u +%FT%TZ) START $task"; cat "$Q/running/$task"
  bash "$Q/running/$task" 2>&1; rc=$?
  echo "=== [$ME] $(date -u +%FT%TZ) END $task rc=$rc"
  dest=done; [[ $rc -ne 0 ]] && dest=failed
  printf '\n# worker=%s rc=%s finished=%s\n' "$ME" "$rc" "$(date -u +%FT%TZ)" >> "$Q/running/$task"
  mv "$Q/running/$task" "$Q/$dest/$task"
done
