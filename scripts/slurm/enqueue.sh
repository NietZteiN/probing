#!/bin/bash
# Add one task to the worker queue:  scripts/slurm/enqueue.sh "python scripts/20_run_model.py ..."
set -euo pipefail
Q="${PROBE_ROOT:-/work/jvl210002/migration/probing}/queue"
mkdir -p "$Q/pending"
name="${2:-task}"
f="$Q/pending/$(date -u +%Y%m%dT%H%M%S)_$(printf '%04d' $((RANDOM % 10000)))_${name}.sh"
printf '#!/bin/bash\nset -uo pipefail\ncd "%s"\n%s\n' "${PROBE_ROOT:-/work/jvl210002/migration/probing}" "$1" > "$f"
echo "queued $(basename "$f")"
