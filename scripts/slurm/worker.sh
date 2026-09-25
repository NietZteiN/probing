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
#
# LANES (2026-09-23). `bash scripts/slurm/worker.sh <lane>` claims only tasks whose name ends
# with `@<lane>` (plus untagged tasks); a worker with no lane claims anything. Needed once two
# workers sit on different cards: an 8B all-position cache belongs on the h100 worker and must
# not be picked up by the a30 one. Tag a task by ending its enqueue name with the lane:
#   scripts/slurm/enqueue.sh "python scripts/20_run_model.py --model llama31-8b ..." e27@h100
#
# UNATTENDED CHAINS (2026-09-24). Three things make a chain of workers safe to leave overnight:
#  - WORKER_IDLE_EXIT_MIN=<n> exits after n idle minutes, so a follow-on worker (submitted with
#    `--dependency afterany:<prev>`) does not hold a GPU once the queue is empty;
#  - a walltime SIGTERM puts the task being run back in pending/ (its script reruns from the
#    start; every stage skips groups whose summary.json exists);
#  - at start-up, a task left in running/ by a worker job that is no longer alive (job 421483
#    timed out mid-task and stranded one) is moved back to pending/.
set -uo pipefail
Q="${PROBE_ROOT:?}/queue"
mkdir -p "$Q"/{pending,running,done,failed}
ME="${SLURM_JOB_ID:-$$}"
LANE="${1:-}"
IDLE_EXIT="${WORKER_IDLE_EXIT_MIN:-0}"
echo "# worker $ME on $(hostname), queue $Q, lane ${LANE:-<any>}, idle exit ${IDLE_EXIT} min"
LOGS="${PROBE_ROOT}/log/slurm"
for t in $(ls -1 "$Q/running" 2>/dev/null); do
  owner=$(grep -h "START $t\$" "$LOGS"/*_worker*.out 2>/dev/null | tail -1 | sed -n 's/^=== \[\([0-9]*\)\].*/\1/p')
  # reclaim only on a confirmed non-RUNNING state; no owner or no answer from sacct = leave it
  state=$([[ -n "$owner" ]] && sacct -X -n -j "$owner" -o State%20 2>/dev/null | awk 'NR==1{print $1}')
  if [[ -n "$state" && "$state" != RUNNING && "$state" != PENDING ]]; then
    mv "$Q/running/$t" "$Q/pending/$t" 2>/dev/null && echo "# reclaimed $t (worker $owner is $state)"
  fi
done
task=""
trap '[[ -n "$task" && -f "$Q/running/$task" ]] && mv "$Q/running/$task" "$Q/pending/$task" && echo "# SIGTERM: returned $task to pending"; exit 143' TERM
idle=0
while true; do
  [[ -f "$Q/STOP" ]] && { echo "# STOP file present, exiting"; break; }
  task=""
  for t in $(ls -1 "$Q/pending" 2>/dev/null); do
    tag="${t%.sh}"; tag="${tag##*@}"; [[ "$tag" == "${t%.sh}" ]] && tag=""   # no @ in the name = untagged
    [[ -n "$LANE" && -n "$tag" && "$tag" != "$LANE" ]] && continue
    task="$t"; break
  done
  if [[ -z "$task" ]]; then
    idle=$((idle+1)); (( idle % 30 == 1 )) && echo "# idle, waiting for work ($(date -u +%H:%M:%S))"
    (( IDLE_EXIT > 0 && idle * 20 >= IDLE_EXIT * 60 )) && { echo "# idle ${IDLE_EXIT} min, exiting"; break; }
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
  task=""
done
