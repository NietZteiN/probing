# Experiment checklist

One line per experiment the paper needs. Status is the truth on disk, not a plan: an item is
ticked only when its numbers are in `results/summary/` and reach the paper through
`scripts/51_tables.py`. Details and numbers: `docs/EXPERIMENTS.md`, `results/NOTES.md`.

*Last updated 2026-10-08. ARR deadline 2026-10-12 (AoE).*

## Figures that explain the paper (2026-10-08)

- [x] Rebuild both papers' main figures to show the task, changed condition, measured
  outcome and finding without relying on the body text. Define added misleading-digit
  errors and separate internal predictions from generated answers. Keep the code paper's
  original-name probe cohort distinct from its fresh screened-name replication.
- [x] Put detailed layer curves and patching plots in the appendix; give full-size
  arithmetic figures independent instructions for reading axes, cells and colours.
- [x] Audit plotted values against saved summaries, inspect the actual-size PDFs and
  keep all text through Limitations on four pages. Numerical results and abstracts are
  unchanged. Verification: `log/figure_understandability_2026-10-08/verification.json`
  in each repository.

## Abstracts for readers outside the project (2026-10-08)

- [x] Keep the direct main-result, evidence and implication structure while explaining
  the task and comparisons in ordinary words. Replace unexplained technical terms with
  concrete descriptions of the models' answers and the separate internal predictors.
  Preserve the central result numbers and scope.
  Verification: `log/accessible_abstracts_2026-10-08/verification.json` in each repository.

## Direct scientific abstracts (2026-10-08)

- [x] Lead both abstracts with the main finding, summarize the design and central
  evidence, and end with the scientific implication. Move illustrative examples and
  secondary results to the introduction and body. Arithmetic centers the matched test
  of resistance to misleading names; code distinguishes recoverable values from written
  values and connects this result to operation-bearing demonstrations.
  Verification: `log/result_first_abstracts_2026-10-08/verification.json` in each repository.

## Direct introduction endings and code abstract (2026-10-08)

- [x] End both introductions with what the experiments show. Rewrite the code abstract
  around one example: define a trace, establish the length-versus-sum conflict, explain
  the separate predictor's correct digit before wrong writes, and connect that observation
  to changing worked examples. Describe the demonstration intervention without introducing
  another numeric example. Scientific caveats remain in Results and Limitations.
  Verification: `log/direct_story_2026-10-08/verification.json` in each repository.

## Contribution and implication (2026-10-07)

- [x] Center the arithmetic abstract on the matched test of resistance to misleading words
  and the code abstract on the separation between recoverable values and trace writes.
  Connect the latter to the worked-example intervention. State what each finding means
  for evaluating or improving reasoning outputs, within the tested tasks. Retain the
  concrete examples and explanation of the extension to prior work. Verification:
  `log/contribution_2026-10-07/verification.json` in each repository.

## Abstract narrative (2026-10-07)

- [x] Replace error labels with concrete descriptions of the answers models produce.
  Give both abstracts an example-to-experiment-to-finding narrative and end with the
  behavioral result. Detailed internal-measurement caveats remain in the body and
  limitations. The text through Limitations still fits four pages in both papers.
  Verification: `log/abstract_story_2026-10-07/verification.json` in each repository.

## Four-page text including limitations (2026-10-07)

- [x] Shorten both papers' main captions and remove repeated explanations in methods and
  related work. Convert limitations to three bullets while retaining scope, probe and
  intervention caveats and the code study's design deviations. All text through Limitations
  now fits four pages; References begins on page five in both papers. Abstracts, introductions,
  results, number macros and result summaries are preserved. Strict checks and visual review
  pass. Verification: `log/concise_2026-10-07/verification.json` in each repository.

## Introduction and explanation (2026-10-07)

- [x] Rewrite both introductions as a story: show a concrete error, pose the question,
  connect it to prior work, and explain what the experiments establish. Define answer-only
  prompting and CoT through the requested output and an evaluated arithmetic example.
  Explain linear probes as predictors of the correct digit from internal activations;
  distinguish the probe's prediction from the model's answer. Revise both abstracts to
  match. Current abstracts: 175/199 words. Both strict builds and all four main-page
  visual reviews pass; numbers and result summaries are unchanged. Verification:
  `log/narrative_2026-10-07/verification.json` and the companion code repository's log.

## Abstract clarity (2026-10-07)

- [x] Rewrite both abstracts around a concrete naming conflict, explain the written
  reasoning steps or code trace, identify the extension of Kudo et al.'s arithmetic study,
  and explain the naming comparison without assuming readers know our terminology.
  Abstracts: 174/192 words. Both strict builds and body layout reviews pass; measured
  results and generated numbers are unchanged. Verification: `log/abstracts_2026-10-07/`.

## Title and terminology (2026-10-06)

- [x] Retitle the arithmetic draft as **Chain of Thought Reduces Errors from Misleading
  Names in Arithmetic**. Define name-suggested values and name errors in both papers;
  replace the old wording in prose, captions, tables and plots. Regenerated number macros
  and result summaries are unchanged. Both strict builds pass within four main pages.
  Verification: `log/terminology_2026-10-06/verification.json`.

## Reviewer follow-ups (2026-10-03)

### Additional targeted validation (round 6)

- [x] R6-A1: independently select code probe layers with five-fold cross-fitting grouped
  by canonical computation; evaluate held-out correct and lure-writing cases for all
  three affected models and demonstration sets 7/11/13.
- [x] R6-A2: screen fresh identifiers on the original seven-model panel, freeze the first
  three passing the 80% meaning gate, then compare trace and expression demonstrations
  on 200 fresh computations in the three affected models and CodeGemma-7B control.
  If no identifier passes, report the replication as unavailable.
- [x] R6-A3: train arithmetic probes on generated neutral chains, with computation-disjoint
  training and validation; evaluate held-out generated chains for both variables in
  OLMo-2-1B-Instruct and Llama-3.2-3B across all three demonstration sets.
- [x] R6-P: validate outputs and report calibrated readouts, uncertainty, exclusions and
  comparisons with the existing results. Keep unvalidated results out of the manuscripts.
- [x] R6-M: integrate all three validated analyses into both ACL ARR drafts, regenerate
  numbers and appendix tables, inspect the rendered pages, and pass both strict builds.

Two one-GPU workers requested for up to 24 hours: **440673**, **440674** (H100/H200).
All fourteen GPU tasks completed with exit code 0 in worker **440673** (50m34s);
the GPU is released. All 24 code replication cells and all twelve generated-chain
test cells are validated. The unused second request (**440674**) was cancelled
before allocation. Collector **440686** exposed a tuple/list comparison after JSON
serialization; normalization fixed it without changing the sample or rerunning any
experiment. Four regression checks and the local collection retry passed.
`results/summary/round6_completion.json` is complete with no errors. All three analyses
are now in the current manuscripts. Code error readouts use independent layer selection;
the screened-name replication retains the original selection-rule deviation. Arithmetic
reports that generated-neutral training still fails OLMo calibration. Both drafts retain
four-page main bodies and pass strict builds. Verification and rendered pages:
`log/paper_update_round6_2026-10-03/`.

CPU preflight and cross-fitting passed as **440676**; arithmetic preparation and precision
guards passed as **440680**, **440684**. The first CPU attempt (**440672**) failed an import
configuration check before any experiment; its logs are preserved. Initial collector: **440686**; the repaired collection completed locally after the
CPU retry (**440950**) waited for resources. Workers release cards after ten idle
minutes. Fixed choices and provenance: `log/round6_2026-10-03/manifest.json`.

### Completed follow-ups (round 5)

- [x] R5-A1: audit generated/cached prefix agreement and report disjoint code probes
  separately on lure writes, correct writes and other errors, with clustered intervals.
- [x] R5-A2: report code effects by identifier, the originally planned panel gate,
  and each model's meaning check; retain the full-data analysis.
- [x] R5-A3: compare arithmetic readouts with task competence, report within-model
  variable differences and leave-one-model-out sensitivity including removal of OLMo-2-1B.
- [x] R5-A4: probe generated arithmetic chains for OLMo-2-1B-Instruct and Llama-3.2-3B,
  both variables and demonstration sets 7/11/13; validate on neutral generated chains.
- [x] R5-A5: test token-count-matched neutral annotations and valid numeric elaborations
  in code demonstrations for OLMo-2-7B and Llama-3.2-3B, sets 7/11/13.
- [x] R5-P: collect and validate follow-ups, update appendix evidence and claims, and
  pass strict builds for both papers.

All five follow-ups are validated and included in the current PDFs. The competence analysis
covers 18 variable pairs from nine models. The GPU workers completed and released their cards;
float32 repairs for both Gemma experiments passed validation, preserving the original caches.
Both strict builds pass with four-page bodies. The round-four collector completed as job
440251 after correcting its summary-file path; all additions are now in the paper. Fixed choices and provenance:
`log/round5_2026-10-03/manifest.json`. The original preregistration is preserved; follow-ups
are exploratory additions. Boxes require measured, validated outputs.

## Readability and new experiment pass (2026-10-02)

- [x] Tighten both drafts: concrete questions, simpler probe/control explanations,
  shorter interpretations and explicit distinctions between association and cause.
  Both strict builds retain a four-page body.
- [x] Run preflight checks: 62 arithmetic CPU tests, 4 tokenizer/probe checks in a CPU job,
  and 42 code CPU tests pass. Across five tokenizers, 2,464 sampled layouts confirm that
  pre-value readouts exclude supplied answer digits; every new cache row is also guarded.
- [x] Inspect both rendered bodies and test the planned appendix layout: four-page bodies,
  no unresolved references, unfilled numbers, missing assets or out-of-page text.
- [x] V1: replace all three code probe runs with disjoint training programs.
- [x] V2: repeat those code probes with demonstration sets 7, 11 and 13.
- [x] G2: add arithmetic value-step probes for Llama-3.2-3B-Instruct,
  Llama-3.1-8B-Instruct and Gemma-3-12B-Instruct (six additional model-variable pairs).
- [x] G3: add Gemma-3-4B-Instruct per-token probes for both variables.
- [x] Collect validated outputs, update claims to match them, regenerate figures and build both papers.

**GPU reservations requested:** two one-GPU H100/H200 workers, jobs **439652** and **439653**,
24 hours each. They began with 13 tasks; the reviewer follow-ups and two cache repairs joined that queue.
Workers exit after 10 idle minutes.
A30/H100 cards were occupied at submission. Workers can use H100/H200 as cards free up;
the slow MIG node is excluded. The fallback reason is recorded in the manifests.
Automatic audit and rebuild: CPU job **439676**, after both workers.
Tokenizer/probe checks: **439688**; pre-value boundary checks: **439757**, both completed.
Submission is not completion: experiment boxes remain open until outputs are validated.
Commands and fixed analysis choices: `log/round4_2026-10-02/manifest.json`.

## Requested completion pass (2026-10-02)

The author requested G1 and C1–C4 below; C5 verifies that their results reach the paper.
G2–G3 were authorized in the subsequent readability/experiment pass and are queued above;
G4 remains outside this paper's scope.

- [x] G1: run OLMo-2-1B-Instruct CoT neutral-twin removal and alternative-lure control.
- [x] C1: regenerate equivalence with preregistered 90% confidence intervals.
- [x] C2: test CoT facilitation and interference against the same equivalence margin.
- [x] C3: correct the matched written-value estimator and pool the exception across all three demonstration sets.
- [x] C4: regenerate missing positional-copy controls for the ladder and Gemma demonstration sets.
- [x] C5: regenerate tables and figures, self-check, and build the paper.

**GPU status:** completed, SLURM job **438941** (`a30,h100,h200`, excluding `g-06-01`;
one-hour limit). A30/H100 GPUs were fully allocated, so H200 backfill was enabled with
the shared `juno` QoS; the reason is recorded in the submission manifest.
G1/C5 outputs collected; final self-check and paper build passed.
Collection/build job **439019** (`dev`, `afterany:438941`) validated all four output files,
refreshed the aggregates, rebuilt the paper and passed the final checks.

**G1 result:** under the gold chain, the intermediate-value readout had 28 baseline lure
errors out of 500 pairs; neutral and alternative-lure patches removed 92.9% and 82.1%.
At the answer there were only 3 lure errors, all removed by both patches. The queried-variable
readout had no baseline lure errors. These controls do not establish a name-specific mechanism
for the exception's freely generated chains. Generated appendix table: `exception_patching.tex`.

**CPU results:** 43/44 CoT lure contrasts equivalent within ±2 points; 61/88 CoT accuracy
contrasts equivalent. Of 15 reliable accuracy contrasts, 5 are also equivalent and 10 are
not. The exception's written-value excess is +3.61 points (95% CI +2.59 to +4.69), pooling
2,739 matched examples across demonstration sets 7, 11 and 13. Missing copy controls now
exist for all 15 requested model/regime cells. Archived `__smoke64` groups are excluded.

**Verification before GPU completion:** data self-check: 0 failures, 80 existing warnings;
paper: 4-page body, no unfilled numbers or unresolved references, A4 portrait, fonts embedded.
Tests: 58 passed, 4 tokenizer-dependent tests skipped; strict paper checks passed.
Per-stage logs are in `log/round3_*.out`; machine-readable status is
`results/summary/round3_completion.json` (complete).

## Done

- [x] Tokenizer rule-4 check and verified word lists (E0)
- [x] Kudo et al. replication on letters: Llama-3.2-3B (E2, E27–E30, t*_eq reproduced) and Llama-3.1-8B (E27–E30, 2026-09-24/25)
- [x] Behaviour, level 3, thirteen models (nine panel models plus four ladder checkpoints), three demonstration sets (E3, E14–E16, E33)
- [x] Behaviour, levels 1–5, Llama-3.2-3B and Llama-3.1-8B, three demonstration sets (E13, E33)
- [x] Gemma-3-4B pretrained, full stack plus demonstration sets 11 and 13 (E15, 2026-09-25); appendix model-kind table
- [x] Model-kind ladder on the Llama-3.1-8B-Instruct spine (Amendment 5); Nemotron demonstration set 7 rerun at full size (2026-09-26: three groups had been a 64-instance smoke run)
- [x] Neutral-trained probes with selectivity control: Llama-3.2-3B (L2–L4), Llama-3.1-8B, Gemma-3-4B-it, Gemma-3-4B, OLMo-2-7B, OLMo-2-1B (E5, E6)
- [x] Patching with both controls, removal and injection (E8–E10, E35); span-by-layer grids for both Llamas (E29)
- [x] No-computation control, level 1 (E34)
- [x] Irrelevant-lure control, level 4, against the matched baseline (E12, `63_irrelevant.py`)
- [x] Free-form chain on instruct models (E36)
- [x] Equivalence bounds, all five levels, every model in the sweep (E37; rerun 2026-09-25)
- [x] Value-written control (`62_value_written.py`)
- [x] Value-step decodability vs behaviour, 10 cells over 5 models (E39)
- [x] Positional-copy control, pooled (E22)
- [x] Lure-distance covariate (E19), teacher-forced lure rate (E20), error taxonomy (E26)
- [x] Identifier-name cue, level 3, four models, three demonstration sets (E38)
- [x] Probe recipe (E17), patching scope (E18), mixed-effects refit (E21), lure index (E23), value-only chain (E31)

## Round 3: open items (2026-09-26)

### GPU experiments

- [x] **G1. Chain-of-thought patching of the exception model (recommended).** OLMo-2-1B-Instruct is the
      only model whose CoT answers follow the name (110–131 lure errors per demonstration set), but its
      earlier CoT patching ran only `inject ctl_word`. The neutral-twin removal and alternative-lure
      controls completed on 2026-10-02. These test the name representation under a force-decoded
      gold chain; baseline lure errors in that readout need not equal errors in the model's own chains.
      Job 438941; collected and checked by job 439019. All caches exist.
      `python scripts/40_patch.py --model olmo2-1b-it --level 3 --regime cot --contrasts main ctl_lure --limit 500`
      (a30 or h100, under 1 GPU-hour). Optional follow-on: the span grid,
      `--grid --sources neutral incongruent_alt --limit 500`.
- [x] **G2. Probes for three more panel models (queued).** The claim that the effect tracks whether
      the value is represented rests on 10 cells from 5 models. Gemma-3-12B-it, Llama-3.1-8B-Instruct and
      Llama-3.2-3B-Instruct have behaviour runs only (`--no-train --no-forced`). A full
      `pipeline.py --stage all` for them adds 6 cells. Roughly 2–4 GPU-hours each; the 12B needs h100/h200.
- [x] **G3. Per-token probes on a non-Llama model (queued).** Llama-3.1-8B makes the queried value
      decodable at the restatement step, two equations before the 3B does. E27+E28 on Gemma-3-4B-it would
      show whether that generalises. About 2 GPU-hours.
- [ ] G4. Instruction-tuned models with their chat template. Not for this paper: it changes token
      positions for the whole design. Stays a Limitation.

### CPU analyses

- [x] **C1. Equivalence at 90% intervals.** Completed 2026-10-02: `57_equivalence.py` recomputes
      preregistered 90% CIs from raw paired observations, clustering on matched set across seeds,
      with 4,000 draws. `equivalence.json` records confidence and provenance; 43/44 CoT lure cells
      remain equivalent. The appendix explains 90% equivalence versus 95% behavioral intervals.
- [x] **C2. Equivalence for accuracy contrasts under CoT.** Completed 2026-10-02: facilitation and
      interference now use the same TOST bound. 61/88 accuracy contrasts meet it; 5/15 reliable
      effects also meet it and 10/15 do not. Generated appendix table: `paper/tables/equivalence.tex`.
- [x] **C3. Written-value control for the exception, pooled.** Completed 2026-10-02: point estimate
      and CI now use the same matched pairs within the same written-value split. `value_written_pooled.json`
      pools seeds 7/11/13 with matched sets as clusters. OLMo L3 v2, value written: +3.61 points
      [+2.59, +4.69], n=2,739 matched observations; corrected not-written estimate: +0.56 points
      [−1.19, +2.45]. The body now cites the three-set pooled estimate through `51_tables.py`.
- [x] **C4. Copy-control coverage.** Completed 2026-10-02: `50_analysis.py` regenerated all three
      CoT regimes for the four ladder checkpoints and Gemma-3-4B (15 model/regime cells).
      Archived smoke groups are excluded from both behavioral and written-value aggregation.
- [x] C5. After any item above: `50_analysis` → `51_tables` → `60_master_figure` → `66_figure_dump` →
      `99_selfcheck` → `make paper` (body ≤ 4 pages, no red `\NUM`, no overfull boxes).

### Author decisions

- [x] **D1. Enforce the preregistered accuracy floor.** The 2026-10-02 verification moved
      OLMo-2-1B's behavioral rows to the appendix. The generator now filters the main table at
      90% mean neutral-CoT accuracy; the data self-check rejects a below-floor row.
- [x] D2. 90% vs 95% (C1): switched to the preregistered 90% equivalence intervals, 2026-10-02.
- [ ] **D3. Author review of the revised manuscript (2026-10-02).** Abstract, introduction,
      results and conclusion now follow behavior → value readout → patching. Secondary controls
      moved to the appendix; `paper/ARGUMENT.md` records the supported claims and their limits.

### Submission

- [ ] ARR responsible-NLP checklist: answers drafted in `docs/ARR_CHECKLIST.md`; confirm the risks (A2) and
      AI-assistant (E1) wording, check against the live form, submit
- [ ] Register a qualified service contributor and verify every author's OpenReview profile;
      contributor registration is due within 48 hours after the submission deadline under the
      [October 2026 ARR policy](https://aclrollingreview.org/cfp).
- [ ] If both papers are submitted concurrently, upload each anonymized companion as concurrent
      supplementary material. Reciprocal anonymous citations and task differences are in both drafts.
- [ ] Final build after the last change; check the PDF uploaded is `paper/main.pdf` at HEAD
- [x] Anonymised: author block, no repository link, no acknowledgements, PDF metadata, no identifying
      `note` in the references (2026-09-25/26)
- [x] Bibliography: all 24 cited entries verified against arXiv / ACL Anthology / NeurIPS / Crossref,
      published versions preferred (2026-09-26)
- [x] Claims audit of body and appendix against `results/summary/` and the raw outputs (2026-09-25/26)
- [x] Limitations re-read against the final results (2026-09-25)
- [x] Verification after the prose rewrite (2026-10-02): result regeneration, data self-checks,
      CPU tests, references and figure roles, strict build and PDF review. Details in `results/NOTES.md`
      and `log/verification_2026-10-02/`.

## Companion code paper: verification follow-up

- [x] Audit saved probe splits. All three models reuse evaluation programs in training;
  the code manuscript now labels its probe scores provisional.
- [x] Fix training selection and add leakage guards while preserving old caches.
- [x] **V1. Rerun code probes on disjoint programs.** OLMo-2-7B, Llama-3.2-3B and
  Llama-3.1-8B; level 5, trace, role v1; three probe seeds and name controls. Use corrected
  `codecue/scripts/30_cache_and_probe.py`. Validate fresh outputs in `probes_disjoint/`,
  then update the numbers generator and mechanism figure to consume them. The held-out
  decodability claim remains unverified until this is complete.

## NAACL review follow-up — 2026-10-08

- [x] Accuracy pass for both papers: regenerate numerical inputs, check saved results and
  figure denominators, correct the code appendix reference and the arithmetic equivalence
  caption, and strengthen the reference checker.
- [x] Readability pass for both papers: explain measurements and controls directly, use
  percentages in the main probe results, and shorten probe-figure headings.

- [x] Center the arithmetic narrative on resistance to a misleading word with the equations
  held fixed; distinguish overall accuracy from answers matching the misleading digit.
- [x] Present probes as a separate investigation; preserve the unresolved internal explanation.
- [x] Distinguish the contributions of the arithmetic and companion code papers.
- [x] Put code error-conditioned probe intervals and computation counts in the main table,
  alongside the independent screened-name replication with separate cohorts labeled.
- [x] Implement and test the prospective code prompt-end versus pre-write comparison.
- [x] Queue two GPU allocations: 449113 (a30) and 449114 (h100).
- [x] Complete GPU runs and validated release: 449113, 449114 and 449115 all completed
  with exit 0. The code repository contains `round7_prompt_comparison.json`. See
  `../codecue/docs/PROMPT_PROBE_COMPARISON.md` and `../codecue/docs/EXPERIMENTS.md`.
- [ ] Integrate the validated comparison into the code manuscript and rebuild/push.

## Not planned for this paper

E7 (instance-level margin link: fitted and reported as unresolved in the appendix), E11,
E24 (already reported as normalised LD), E32 (b)–(c) (operand-copy error; superseded by E33).
