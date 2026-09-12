#!/usr/bin/env bash
# Build the probing uv environment: obtune's exact lock + torch cu129 overlay + analysis extras,
# in a SEPARATE venv so obtune's and bidirectional's envs (which their live SLURM jobs use) are
# never touched. RUN this, do not source it.
#   bash env/setup_env.sh
set -euo pipefail
PROBE_ENV="${PROBE_ENV:-/work/jvl210002/migration/envs/probe-cu129}"
OBTUNE_ROOT="${OBTUNE_ROOT:-/work/jvl210002/migration/obtune}"
LOCK="$OBTUNE_ROOT/env/lock-obtune.txt"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRAS="$HERE/extras.txt"
export TMPDIR="${TMPDIR:-/work/jvl210002/migration/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TMPDIR/uv-cache}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$(dirname "$PROBE_ENV")"
UV="${UV:-$(command -v uv)}"
[[ -d "$PROBE_ENV" ]] || "$UV" venv "$PROBE_ENV" --python 3.12
PY="$PROBE_ENV/bin/python"
echo "[1/3] replaying $LOCK ($(wc -l < "$LOCK") packages)"
"$UV" pip install --python "$PY" -r "$LOCK"
# juno's driver is r550 (CUDA 12.4); the lock's cu130 torch reports cuda.is_available()==False.
echo "[2/3] overlaying torch 2.11.0+cu129"
"$UV" pip install --python "$PY" \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129"
echo "[3/3] extras under lock constraints"
"$UV" pip install --python "$PY" -c "$LOCK" -r "$EXTRAS"
"$PY" - <<'PYEOF'
import importlib.metadata as m
for p in ["torch","transformers","accelerate","scikit-learn","statsmodels","matplotlib","seaborn","pandas","numpy","scipy","pytest","pypdf","pyyaml"]:
    print(f"{p:14s} {m.version(p)}")
PYEOF
echo "ENV BUILD OK: $PROBE_ENV"
