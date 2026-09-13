# Experiment registry

*Written 2026-09-12. One row per experiment the paper needs; each has a claim it serves
(paper/ARGUMENT.md §1), a command, a partition, a cost, and a status. Update the status column
as jobs land; never add a result to the paper that has no row here.*

Cost basis: 3B on A30 ≈ 1 GPU-h per (model, level) `run` stage; 8B on H100 ≈ 2 GPU-h.
Probes ≈ 1 GPU-h per (model, level, regime, train split). Patching ≈ 1.5 GPU-h per (model,
level, regime). Storage ≈ 6–8 GB per (model, level) on scratch.

## A. Gates (must pass before anything is claimed)

| id | experiment | serves | command | where | cost | status |
|---|---|---|---|---|---|---|
| E0 | Tokenizer rule-4 check per model: number words and neutral words single tokens in five contexts; template layout dump | Appendix A; every twin comparison | `make tokcheck` | dev (CPU) | 5 min | **done** for llama32-3b, llama32-3b-it, llama31-8b, llama31-8b-it (95/112 words); pending for gemma3-4b, olmo2-7b-it, olmo3-7b-think |
| E1 | Smoke run: 64 instances × 4 groups × 2 regimes on Llama-3.2-3B; layout guard, parse rate, accuracy sanity | nothing in the paper; catches format bugs | `20_run_model.py --model llama32-3b --level 3 --groups neutral incongruent@v2 congruent@v2 letter --limit 64` | h200 | 1 min | **done** 2026-09-12 (job 391545): 0 excluded; CoT acc 94–98% (n=64/group); direct acc 14–27%, i.e. near floor for the 3B base model (it answers `name=0` most of the time). Outputs moved to `runs/llama32-3b/L3_smoke/` so the full run does not skip them |
| E2 | Kudo replication: probes trained and tested on letters, L3, CoT, Llama-3.2-3B; report Acc≺CoT, Acc≻CoT and first-above-τ per variable against their Table 3 (v1 at equation 5, v2 at 2; τ=0.9) and Table 7 accuracy (93.15%) | Results ¶1 "Replication" | `30_train_probes.py --model llama32-3b --level 3 --regime cot --train train_letter --test-groups letter` then `50_analysis.py` → `replication.json` | h200 | 3 GPU-h (unbatched) | **done** for llama32-3b (job 391548): first position above τ = the value step for both variables, pre-CoT max .17/.48 vs Kudo .18/.33 — reproduced |

## B. Core (level 3, two core models, both regimes; the go/no-go on Oct 1 is decided on these)

| id | experiment | serves | command | where | cost | status |
|---|---|---|---|---|---|---|
| E3 | Behaviour: free greedy generation, all 10 groups (letter, neutral, congruent/incongruent/incongruent_alt/neutral_alt × {v1, v2}), both regimes; accuracy, lure rate, interference, facilitation, CoT-protection contrast; cluster-bootstrap CIs; clustered logit `correct ~ condition × regime` | C1, C2 → Table 1 | `pipeline.py --stage run --models llama32-3b llama31-8b` (also caches hidden states for E4) | h200 (a30/h100 full) | 1 + 2 GPU-h | **llama32-3b done** 2026-09-12 (job 391546, 14 min, 0 excluded): CoT neutral 96.6, incongruent@v1 94.6 (interference +2.0 [1.1, 2.9], lure rate 0.4% vs pseudo-lure 0.1%), incongruent@v2 99.4 (a number-word name on the intermediate HELPS, congruent or not: 99.4–99.5 vs 96.6); direct neutral 13.4 (floor), lure rate 8.1–8.6% vs pseudo-lure 4.7%, congruent +4.6/+5.5. llama31-8b submitted |
| E4 | Hidden-state cache at 13 labelled positions, all layers, for 10k neutral-train + 10k letter-train + 20k test instances, both regimes | E2, E5, E6, E7 | same job as E3 | same | included | **llama32-3b done** (≈100 instances/s on H200; 5.5 GB per regime) |
| E5 | Neutral-trained probes: per (role, position, layer, seed ∈ {0,1,2}); evaluated on every group; accuracy, lure rate, lure mass, margin; crossover step (first position where layer-max margin > 0 and stays > 0); FDR over layer × position | C3 → Figure 2, crossover | `pipeline.py --stage probes` (train_neutral) | h200 | 3 GPU-h unbatched; minutes batched | **done** for llama32-3b (391547/391550); 8B queued (391916–391919). See results/NOTES.md |
| E6 | Selectivity: Hewitt–Liang control probe (label = hash of the target's name) at every cell; report accuracy − control; cells below 0.3 are not interpreted | gate for E5 cells; Appendix B | included in E5 (`control=True`) | same | included | not started |
| E7 | Instance-level link: `lure_error ~ margin(end@r) + margin(query) + margin(cotpre@r)` at the best neutral-accuracy layer, clustered on set | C3 → RQ2 last sentence | `50_analysis.py` → `link.json` | login | minutes | not started |
| E8 | Patching `main`: neutral → incongruent at every token carrying the target name; single layers, 4-layer windows, all layers; read next token at `cotpre@r` and `anspre`; recovery = P(correct after \| lure error before) | C4 → Figure 3 | `pipeline.py --stage patch` | h200 | 1.5 GPU-h per (model, regime) | **done for both core models with controls** (391920–391923). 3B direct: neutral patch removes the lure 52%/39% vs alt-lure 35%/18%, word-control damage 17–22% (threshold fails at floor accuracy). 8B direct: neutral and alt-lure patches remove it equally (43%/31%), follows-new-lure at base rate, damage 9–10%: no name-caused lure effect. CoT: no lure errors, controls clean. See results/NOTES.md |
| E9 | Patching control `ctl_word`: neutral_alt → neutral; damage = P(wrong after \| correct before); must be < 5% at the reported layer set | C4 validity | same job as E8 | same | included | **done**: 3B direct 17–22%, 8B direct 9–10% (both above 5%); CoT 0% |
| E10 | Patching control `ctl_lure`: incongruent_alt → incongruent; follows-new-lure rate | C4 positive control | same job as E8 | same | included | **done**: follows the new lure 7–9% (3B) / 4% (8B), at base rates |
| E11 | Target position: every behavioural and probe quantity split by target = intermediate (v2) vs queried (v1) | Table 1 rows; PLAN rule 5 | by construction of the groups; `51_tables.py` | login | — | not started |

## C. Controls and robustness (after the gate; appendix unless the effect is large)

| id | experiment | serves | command | where | cost | status |
|---|---|---|---|---|---|---|
| E12 | Irrelevant-lure control at level 4: number word on the distractor; lure rate and probe lure mass vs incongruent@v2 | "binding vs presence" sentence in RQ1 | `pipeline.py --stage all --levels 4` | a30 / h100 | 3 GPU-h per model | not started |
| E13 | Levels 2 (in order) and 5 (three steps): behaviour + probes | robustness of C1–C3 across dependency structure and depth | `pipeline.py --stage run,probes --levels 2 5` | a30 / h100 | 4 GPU-h per model | not started |
| E14 | Base vs instruct pair: llama32-3b-it and llama31-8b-it, level 3, full stack | "tuning" row of Table 1; Limitations | `pipeline.py --stage all --models llama32-3b-it llama31-8b-it` | a30 / h100 | 5 + 8 GPU-h | not started |
| E15 | Second family: Gemma-3-4B (pt), level 3; requires download, E0, and a loading check for the multimodal checkpoint (`runner.load_model`) | family robustness | E0 then `pipeline.py --stage all --models gemma3-4b` | a30 | 5 GPU-h | blocked on download + E0 |
| E16 | Reasoning-tuned model: OLMo-3-7B-Think (availability and chat-template behaviour to verify); if unavailable, OLMo-2-7B-Instruct as the AI2 family row | "base vs reasoning-tuned" pair in PLAN §4.5 | E0 then `pipeline.py --stage all --models olmo3-7b-think` | h100 | 8 GPU-h | blocked on availability |
| E17 | Probe recipe robustness: lbfgs (scikit-learn) and standardized inputs vs Kudo's SGD; same picture required | Appendix B | `30_train_probes.py --optimizer lbfgs [--standardize]` on llama32-3b | a30 | 1 GPU-h | not started |
| E18 | Patching scope: prompt-only name positions vs all occurrences | Appendix C | `40_patch.py --scope prompt` on llama32-3b | a30 | 1.5 GPU-h | not started |
| E19 | Lure-distance covariate: interference and margin as a function of \|lure − true\| | Appendix; one sentence in RQ1 if monotone | `50_analysis.py` (add-on; data already in `behavior.csv`) | login | minutes | not started |
| E20 | Teacher-forced lure rate (does the model write the lure at `cotpre@r` when the chain so far is correct?) vs free-generation lure rate | RQ1 footnote; separates "the chain derails early" from "the readout follows the name" | from `forced_logits.jsonl` (E4); analysis add-on | login | minutes | not started |
| E21 | Mixed-effects logistic regression (random intercept per matched set) beside the clustered logit | Appendix D; reviewer question | `stats.mixed_logit` on `behavior.csv` | login | minutes | not started |
| E22 | Positional-copy control (Liu 2026): among lure errors at P5 in the CoT regime, the share whose trailing chain number was the TRUE value (copying cannot explain the error) vs the lure (it can); also the P5 lure rate conditioned on a correct chain up to P4 | validity of the "late readout" cell; RQ3 paragraph | from `behavior.jsonl` generations + `forced_logits.jsonl`; analysis add-on | login | minutes | not started |
| E23 | Geirhos-style lure index = lure answers / (lure + true answers), congruent trials excluded, reported beside accuracy | Table 1 column | `50_analysis.py` add-on | login | minutes | not started |
| E26 | Error analysis of the number-word-name effect in the CoT regime: why does a number-word name on the INTERMEDIATE raise accuracy (99.4 vs 96.6, congruent and incongruent alike) while on the QUERIED variable it lowers it (95.1/94.6 vs 96.6, again regardless of congruence)? Classify neutral-condition errors (copy of the wrong operand, arithmetic slip, chain format) vs number-word errors; check whether the effect is a property of the word class (number words as names) rather than of the value | interpretation of Table 1; may become a finding | analysis over `behavior.jsonl` generations | login | hour | not started |
| E24 | Zhang–Nanda normalised logit difference for patching, (LD_patched − LD_inc)/(LD_neu − LD_inc) with LD = logit(true) − logit(lure), beside recovery rate | Figure 3 second panel / appendix | `40_patch.py` (record logits) + analysis | in E8 | included | not started |

### Risk found by E1: the direct regime is at floor for Llama-3.2-3B

With three demonstrations the 3B base model answers 14–27% of level-3 problems correctly
without a chain, mostly by writing `name=0`. Interference cannot be measured on a floor. Two
options, to be decided after the full run (E3) confirms it on n=2000: (a) report the direct
regime for this model as a floor and rely on Llama-3.1-8B for the regime contrast; (b) add a
pre-registered amendment giving the direct regime more demonstrations (Kudo et al. used 50 for
their "general prompting" variant) via a `--n-demos` option, keeping the CoT regime at three.
Option (b) changes the token layout of the direct regime only and does not touch any twin
comparison.

## F. Kudo et al.'s figures and tables, reproduced in the cue-conflict setting

*Added 2026-09-12 at the lead's request. Kudo et al. report four things we can produce for every
condition: a per-token probing heatmap with a max-over-layers curve (their Fig. 2), the
t*/t*_eq and Acc≺CoT/Acc≻CoT table (their Tables 2–3), an equation × 4-layer-window patching
grid (their Figs. 5–6), and probe-prediction trajectories on wrong instances (their Fig. 3).
Our versions put neutral, congruent and incongruent side by side and add the lure: a lure-rate
heatmap, a margin heatmap, t_lure (last position where the probe still reads the lure), and
patching sources that are the matched twins rather than a different problem.*

| id | experiment | serves | command | where | cost | status |
|---|---|---|---|---|---|---|
| E27 | All-token hidden-state cache: every token of the instance region (input + output; ~50 tokens at level 3), layer stride 2, for neutral/letter probe-train (4,000 each) and the test groups letter, neutral, congruent@{v1,v2}, incongruent@{v1,v2} (2,000 each), both regimes, core models. ~55 GB per (model, regime) on scratch | E28–E30 | `20_run_model.py --all-positions --layer-stride 2 --train-limit 4000 --groups ...` → `<group>__alltok/` | h200 | 1 GPU-h per model | not started |
| E28 | Kudo Fig. 2 analogue: heatmap of probe accuracy over token position × layer for each variable, with the max-over-layers curve above it, one column per condition (neutral, congruent, incongruent); below it, for incongruent, the lure-rate heatmap and the margin heatmap on the same axes; x-axis labelled with the actual tokens of the running example. Kudo Tables 2–3 analogue per condition: t*, t*_eq, Acc≺CoT, Acc≻CoT for the true value, plus t_lure and max lure rate pre/post CoT | Figure 2 (main) and Appendix B (full) | `30_train_probes.py --suffix __alltok` then `53_kudo_figs.py --model M --level 3` | h200 (probes, batched) + login (figures) | 1 GPU-h per (model, regime) | not started |
| E29 | Kudo Figs. 5–6 analogue: patch whole equation spans (each input equation, the query, each chain step) × 4-layer windows; three sources: (i) a different neutral problem with a different answer (their design; replication of the recency-bias grid in our format), (ii) the matched neutral twin (lure removal / normalised LD), (iii) the matched alternative-lure twin (follows the new lure); read tokens: the target's value step and the final answer. Rendered as their grid with the max-over-window curve under the token axis | Figure 3 alternative; Appendix C | `40_patch.py --grid --sources other neutral incongruent_alt` | h200 | 2 GPU-h per (model, regime) | not started |
| E30 | Kudo Fig. 3 analogue: per-token top-1 probe predictions (best layer per position) on (a) the lure-error instances, (b) a random sample of incongruent instances, colour-coded true / lure / other, to show where the lure enters and leaves the readout | Appendix; one panel may replace Figure 1's sketch | `53_kudo_figs.py --trajectories` | login | minutes | not started |
| E31 | Kudo Table 6 analogue: the "Simple CoT" format (only sub-results, `cup=5, pen=6`) as a third regime between direct and full chain, behaviour + labelled-position probes | Appendix; tests whether restating the name beside the value is what removes the lure | `20_run_model.py --regimes simple` (new format in generator) | h200 | 1 GPU-h per model | not started |

## D. Not run, stated in Limitations

Optimised or searched lures; digit-bearing names (`x2`); non-English number words; real code;
models above 8B; nonlinear or sparse probes; free-generation patching (the protocol reads the
next token under teacher forcing, as Kudo et al. do).

## E. Order of execution

1. E1 (queued) → fix anything it shows.
2. E3+E4 for llama32-3b (one job), then E2 and E5+E6 (three jobs), E8–E10 (two jobs). Same for
   llama31-8b on h100. This is the Oct 1 gate: E3 interference CI excludes 0 in at least one
   regime, E5 places the lure somewhere with selectivity ≥ 0.3, E9 damage < 5%.
3. E7, E11, E19–E23 are analysis-only and run as soon as E3–E6 exist; E24 needs the logits recorded in E8.
4. E12, E13 on both core models; E14 on the pair; E17, E18 on llama32-3b.
5. E15, E16 only if the gate passed with time to spare (Oct 2–8).

Total for the gate: ≈ 12 GPU-h. Everything in C: ≈ 50 GPU-h more.
