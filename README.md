# Cue conflict in chain-of-thought arithmetic

Does chain of thought let a language model's computation override a misleading variable name,
and where inside the model does that happen? A Stroop-style test built on Kudo et al.'s
(Findings of EACL 2026) controlled multi-step arithmetic: every problem appears with
**neutral**, **congruent** and **incongruent** variable names under identical arithmetic, with
and without chain of thought. Linear probes trained on neutral problems only read out whether
the model represents the true value or the lure; activation patching from the matched neutral
twin tests whether the name's influence causes the errors.

- [`PLAN.md`](PLAN.md) — the science: thesis, conditions, positions, metrics, outcome table,
  timeline to the **October 12, 2026** ARR deadline (short paper).
- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) — the execution: what runs, on what
  partition, in what order, with cost estimates and go/no-go gates.
- [`PREREGISTRATION.md`](PREREGISTRATION.md) — contrasts, thresholds and exclusions fixed
  before any model is run. Amend by dated appendix only.
- [`RELATED_WORK.md`](RELATED_WORK.md) — every paper positioned against, with what was read.
- [`CLAUDE.md`](CLAUDE.md) — operating rules for working in this repo on juno.
- [`paper/`](paper/) — ACL/ARR short-paper draft (`make paper`).
- `third_party/faithful-cot-multistep-arithmetic/` — Kudo et al.'s public code, cloned
  read-only for reference (their format and probe recipe are re-implemented in `src/cueconf`;
  their Lightning + patched-transformers stack does not run on this cluster's toolchain).

## Layout

```
src/cueconf/     generator · prompts (positions P1–P5) · tokcheck · runner · probes · patching · stats
scripts/         10_build_dataset  11_tokenizer_check  20_run_model  30_train_probes  40_patch
                 50_analysis  51_tables  52_figs  00_status  92_page_budget   slurm/{submit,pipeline}.py
configs/         models.yaml (panel) · dataset.yaml (sizes, seeds) · compute.yaml (partitions)
tests/           generation rules 1–3 and 6, chain format, positions, env hygiene
data/            built by 10_build_dataset.py (gitignored; manifest carries seeds + content hash)
results/summary/ small aggregates from 50_analysis.py (committed); raw outputs live on scratch
paper/           main.tex, refs.bib, acl.sty, generated tables/ and figures/
papers/          the PDFs this paper positions against, including Kudo et al.
```

## Quick start (juno)

```bash
source scripts/env.sh                      # PROBE_* roots, env on PATH, HF cache on scratch
make test                                  # 41 CPU tests, fine on the login node
make tokcheck                              # rule-4 word lists per model (dev partition job)
make data                                  # datasets for levels 3, 2, 5, 4 (after tokcheck)
python scripts/slurm/pipeline.py --stage all --models llama32-3b llama31-8b
python scripts/00_status.py                # what exists
python scripts/50_analysis.py && python scripts/51_tables.py && python scripts/52_figs.py --model llama32-3b
make paper                                 # tectonic; prints body page count vs the 4-page limit
```

The running example: `pen=1+two, two=2+3; pen=?` — `two` has true value 5 and lure value 2.

## Task format

Kudo et al.'s levels (1–5), digits 0–9, `+`/`−`, all values single digits, three fixed
same-level demonstrations, one problem per line:

```
pen=1+cup, cup=2+3; pen=?
pen=1+cup, cup=2+3, cup=5, pen=1+cup, pen=1+5, pen=6        (chain of thought)
pen=6                                                        (direct answer)
```

Conditions per matched set: `letter`, `neutral`, `congruent@v`, `incongruent@v`,
`incongruent_alt@v`, `neutral_alt@v` for each chain variable `v`, plus `irrelevant@v3` at
level 4. Generation rules and what enforces them are in `src/cueconf/generator.py` and
`tests/test_generator.py`.
