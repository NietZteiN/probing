# Experiment plan (execution)

*Written 2026-09-12 against PLAN.md §12. Deadline: ARR October 12, 2026 (AoE); go/no-go Oct 1.*

## 0. Reproduction gate (Sep 12–17)

| step | command | partition | wall | output |
|---|---|---|---|---|
| tokenizer check | `make tokcheck` | dev (CPU) | 5 min | `data/words/<model>.json` |
| build data | `make data` | login | 30 s | `data/L{3,2,5,4}/` |
| smoke run | `20_run_model.py --model llama32-3b --level 3 --limit 64` | a30 | 10 min | layout guard, accuracy sanity |
| full run | `pipeline.py --stage run --models llama32-3b` | a30 | ~1.5 h | caches, behaviour |
| Kudo replication | `30_train_probes.py --train train_letter --test-groups letter` | a30 | ~40 min | t*, Acc≺CoT/≻CoT for Table 3 row |

Gate: Llama-3.2-3B letter-condition CoT accuracy ≈ 93% (Kudo Table 7: 93.15) and t*_eq(v1) =
5, t*_eq(v2) = 2 with τ = 0.9 (Kudo Table 3). If the format or positions are off, this is where
it shows.

## 1. Core campaign (Sep 18–Oct 1)

Per model: one `run` job (both regimes, all groups) → two `probes` jobs per regime
(train_neutral, train_letter) → one `patch` job per regime. Level 3 first; then levels 2, 5, 4.

Cost per model and level (estimated for 3B on A30; ×2 for 8B on H100):

| stage | what | est. |
|---|---|---|
| free generation | 2 regimes × ~16k instances × ≤96 tokens | 30 min |
| forced caching | 2 regimes × ~36k forward passes, hidden states at 11 positions | 30 min |
| probes | 2 regimes × 2 roles × 5 positions × 29 layers × 3 seeds × (task + control) | 60 min |
| patching | 2 regimes × 3 contrasts × 2 targets × 2000 pairs × 37 layer sets | 90 min |

Storage per model and level: ~6 GB (3B) / ~8 GB (8B) on scratch.

Two core models at level 3: ≈ 8 GPU-h. Full panel (7 models) × 4 levels: ≈ 100 GPU-h, well
inside a30 + h100 capacity; the constraint is wall-clock for analysis, not GPU.

## 2. Analysis (continuous)

`50_analysis.py` → `results/summary/`; `51_tables.py` → Table 1 + `numbers.tex`;
`52_figs.py` → Fig 2 (margin heatmap CoT vs direct) and Fig 3 (patching by layer with controls).
Fig 1 (three twins + margin sketch) is drawn by hand in TikZ from a real instance.

## 3. Go/no-go (Oct 1)

Submit if, on both core models at level 3: (a) interference is non-zero with a CI excluding
0 in at least one regime; (b) the crossover analysis places the lure somewhere specific
(either fading or persisting) with selectivity ≥ 0.3; (c) `ctl_word` damage < 5%. Otherwise
move to the next ARR cycle with the same pre-registration.

## 4. Extensions if time allows (Oct 2–8)

Base-vs-instruct pair (Llama-3.2-3B-Instruct), a second family (Gemma-3-4B; needs download and
a loading check for the multimodal checkpoint), a reasoning-tuned model (OLMo-3-7B-Think;
availability to verify), levels 2 and 5, lure-distance covariate.

## 5. Not doing

Optimised lures, real code, non-English number words, models above 8B, sparse or nonlinear
probes. All named in Limitations.
