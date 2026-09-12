# Sourced by every sbatch script and by `make`. Roots are defined ONCE here.
#
# EVERY name is prefixed (PROBE_*). The cluster exports SCRATCH=/scratch/$USER, which this
# account cannot write; an unprefixed `SCRATCH="${SCRATCH:-...}"` keeps the cluster's value and
# sends HF_HOME to an unwritable path on every compute node while working on the login node
# (bidirectional job 388948, 2026-09-10). tests/test_env.py asserts no unprefixed exports.
export PROBE_ROOT="${PROBE_ROOT:-/work/jvl210002/migration/probing}"
export PROBE_ENV="${PROBE_ENV:-/work/jvl210002/migration/envs/probe-cu129}"
export PROBE_SCRATCH="${PROBE_SCRATCH:-/work/jvl210002/migration}"
# Hidden states, probe weights and patching sweeps: tens of GB per model, disposable,
# regenerable from the manifests. /scratch/juno/<user> (29 TB free, WekaFS, 11x faster reads
# than /work). NB the cluster's own $SCRATCH points at /scratch/<user>, which does not exist.
export PROBE_OUT="${PROBE_OUT:-/scratch/juno/jvl210002/probing}"
# Models: the shared hub cache on scratch (obtune moved it there 2026-09-10; token lives inside).
export PROBE_HF_STORE="${PROBE_HF_STORE:-/scratch/juno/jvl210002/hf_home}"

export PATH="$PROBE_ENV/bin:$PATH"
export PYTHONPATH="$PROBE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PROBE_HF_STORE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TMPDIR="${TMPDIR:-$PROBE_SCRATCH/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PROBE_SCRATCH/cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROBE_SCRATCH/cache/triton}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

# Fail loudly rather than three frames into a download.
for _d in "$HF_HOME" "$TMPDIR" "$PROBE_OUT"; do
  if ! mkdir -p "$_d" 2>/dev/null || [ ! -w "$_d" ]; then
    echo "FATAL: $_d is not writable -- check PROBE_* in scripts/env.sh" >&2
    return 1 2>/dev/null || exit 1
  fi
done
unset _d
