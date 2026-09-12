# Operating rules — `probing/`

*Last updated: 2026-09-12.*

This project runs a cue-conflict (Stroop) test of chain-of-thought arithmetic. The science is
[`PLAN.md`](PLAN.md); the execution is [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md);
what is fixed before results exist is [`PREREGISTRATION.md`](PREREGISTRATION.md). This file is
how to work in the repo. It does not inherit `obtune/CLAUDE.md` or `bidirectional/CLAUDE.md`,
but the cluster facts below are the same ones those files learned the hard way.

## 1. Compute (juno, SLURM 24.11.5, no GPU on the login node)

- **The login node caps virtual memory at 8 GB** (`ulimit -v 8388608`). Loading a tokenizer or
  model, or importing scipy under transformers, dies there and works in a job. `make test` is
  CPU-only and fits; anything with a tokenizer goes through `sbatch` (`dev` partition, CPU, 2 h).
- **Nothing here trains a model.** bf16 inference of an 8B model is 16 GB, so every GPU stage
  fits on `a30` (24 GB, ≤4B models) and `h100` (8B). Neither carries a QoS, so this project
  **never competes for the 4-job `juno` pool** on `h200`/`normal` that obtune and bidirectional
  share. Do not submit to h200 without a reason written in the job manifest.
- Exclude `g-06-01` on h100 (MIG slices, 2.8× slower). `configs/compute.yaml` does this.
- Jobs skip groups whose `summary.json` exists, so a killed job resumes by resubmission.

## 2. Environment

- `/work/jvl210002/migration/envs/probe-cu129`: obtune's lock replayed + torch `2.11.0+cu129`
  (juno's driver is r550; the lock's cu130 torch sees no GPU) + `env/extras.txt` under `-c` the
  lock. **Never add a dependency to obtune's lock**; new deps go in `env/extras.txt`.
- `scripts/env.sh` exports only `PROBE_*`-prefixed roots (plus HF/TMP cache variables). The
  cluster exports `SCRATCH=/scratch/$USER`, which is unwritable; an unprefixed name silently
  keeps the cluster's value on compute nodes. `tests/test_env.py` enforces the prefix.
- Models live in the shared hub cache `/scratch/juno/jvl210002/hf_home` (the token is inside
  it; moving `HF_HOME` without it breaks every gated Llama/Gemma repo with a 401). Llama-3.2-3B
  base and instruct were downloaded there 2026-09-12; Llama-3.1-8B was already present.

## 3. The rules that make a result mean something

Each is enforced by a test or a runtime guard, because each is a way to produce a plausible,
wrong table.

1. **Exactly one number-word name per instance** (rule 1) and **the lure is never a value or
   an operand digit in the problem** (rule 2, stronger than the plan's wording so a lure can
   never be "right for another reason"). `tests/test_generator.py`.
2. **Twins are byte-identical up to one name slot** (rule 6); every comparison is within a
   matched set and every CI is a cluster bootstrap over sets. `tests/test_generator.py`,
   `stats.bootstrap_ci`.
3. **Rule 4 is decided by the tokenizer, not by eye.** `11_tokenizer_check.py` writes the words
   that are one token in every context, per model; `10_build_dataset.py` uses the intersection
   and records `word_pool.verified`; `20_run_model.py` refuses an unverified dataset; the runner
   excludes any instance whose token layout differs from the group's and fails above 1%.
4. **Probes are trained on neutral data only.** Training on incongruent data would let the
   probe learn the lure. `train_letter` exists solely for the Kudo replication.
5. **P1 (the name token) is never evidence.** It is cached and plotted as a reference only.
   Claims are made at `end@r` (P2), `query` (P3), `cotpre@r` (P4), `anspre` (P5).
6. **Probe positions are absolute token indices** (fixed demos + rule 4), as in Kudo et al.;
   `summary.json` records them and the guard in (3) keeps them constant within a group.
7. **Patching sources and destinations must have identical token layouts**; asserted per pair.
   Recovery is reported only alongside the two control patches (`ctl_word`, `ctl_lure`).
8. **Selectivity is reported with every probe accuracy** (Hewitt & Liang control task keyed on
   the name word), and every probe has ≥3 seeds.
9. **No number in the paper is hand-typed.** `51_tables.py` writes `paper/numbers.tex` and
   `paper/tables/`; an unfilled `\NUM{}` renders red. `make numbers` lists what is outstanding.

## 4. Wording (PLAN.md §7)

"Misleading" or "incongruent", never "adversarial". "Interference" and "facilitation" in the
Stroop sense. Claims stay on arithmetic with variable assignment; code is motivation.

## 5. Destructive commands — human in the loop

Never `rm -rf` or bulk-delete without stating the blast radius, dry-running with `ls`/`find`,
and waiting for explicit approval. Hidden-state caches are hours of GPU time; `data/` is
seconds but its content hash is what the manifests and the paper cite.

## 6. Storage and provenance

Caches, probes and patching sweeps go to `$PROBE_OUT = /scratch/juno/jvl210002/probing` (purge
policy unconfirmed; everything there is regenerable from `data/` + manifests). Aggregates go to
`results/summary/` (committed). Every SLURM job writes `log/slurm/<jobid>_<name>.json` with
argv, git sha, node and exit code; every run directory carries `summary.json`.
