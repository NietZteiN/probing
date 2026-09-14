#!/bin/bash
# Add one task to the worker queue:  scripts/slurm/enqueue.sh "python scripts/20_run_model.py ..." [name] [needs-path]
# The optional third argument is a path that must exist before the task may start (a cache
# directory, a summary.json); the worker defers the task until it does. Added 2026-09-14 after
# three probe jobs ran before their cache jobs and failed.
set -euo pipefail
Q="${PROBE_ROOT:-/work/jvl210002/migration/probing}/queue"
mkdir -p "$Q/pending"
name="${2:-task}"
f="$Q/pending/$(date -u +%Y%m%dT%H%M%S)_$(printf '%04d' $((RANDOM % 10000)))_${name}.sh"
printf '#!/bin/bash\n' > "$f"
[[ -n "${3:-}" ]] && printf '# needs: %s\n' "$3" >> "$f"
printf 'set -uo pipefail\ncd "%s"\n%s\n' "${PROBE_ROOT:-/work/jvl210002/migration/probing}" "$1" >> "$f"
echo "queued $(basename "$f")"
