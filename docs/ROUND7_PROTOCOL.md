# Paired prompting effects and response controls

Fixed on 2026-10-09 before any new format or final-query outputs are collected.
These are exploratory reviewer follow-ups; they do not amend the original registered claims.

- [ ] R7-A: Recompute CoT-minus-answer-only excess name errors on exactly the 44 existing
  equivalence cells. Pair neutral/misleading and direct/CoT versions within matched set
  and demonstration seed. Cluster all demonstration repeats by matched set. Report 95%
  intervals, neutral and misleading accuracy changes, and reductions with the same sign
  in all three seeds and an interval below zero. Retain the existing equivalence results.
- [ ] R7-B: Cross variable-name repetition with numeric equations in demonstrated outputs.
  Four formats show named/unnamed numeric equations or named/unnamed values, in dependency
  order, followed by a common `Answer: <digit>`. All input programs and demonstration
  identities are held fixed. Ordinary-word notes match demonstration-head token counts;
  actual generated lengths are measured, not forced to match. Use 200 level-3 matched
  sets selected by a fixed hash, retaining only one rendering per arithmetic computation.
  Both target roles and three demonstration sets (7, 11, 13) are tested in Llama-3.2-3B,
  Llama-3.1-8B and OLMo-2-1B-Instruct. Neutral-only calibration uses 100 independent
  probe-training computations, disjoint from every test and demonstration computation.
  Try 3, 8, then 16 examples; select the first common count at which every format has
  at least 90% calibration accuracy and their accuracy range is at most two points.
  If no count passes, use 16 and disclose failed matching. Confirm on held-out neutral
  tests: a comparison is competence-matched only if both accuracies are at least 90%
  and the paired 90% accuracy-difference interval lies within +/-2 points. Report all
  comparisons, including failures. Filler and format differences remain limitations.
- [ ] R7-C: Resume after an actual, completely correct generated calculation. Select paired
  ordinary/misleading runs from existing CoT generations only when every generated
  equation, substitution and value matches the executable gold calculation. Freeze up
  to 200 independent computations per model/seed/target before observing continuations.
  Cross a final query repeating the queried name versus requesting the original value
  without its name, with a digit-only versus prefilled named-assignment output.
  Within each naming condition, the input and complete generated calculation are identical
  across the four continuations. The correct queried value is already present. Compare
  excess final name errors, correct answers and parse coverage. This is conditional on
  successful original calculation and does not estimate the effect of making it correct.
  Tail lengths and output markers vary by design; no causal claim about a single feature.
- [ ] Validate outputs, update the paper and submission bundle, check the four-page limit,
  and push the completed artifacts.

Use greedy decoding, equal generation budgets within each experiment, full-vocabulary
outputs, and bootstrap arithmetic-computation clusters across names and demonstration
repeats for R7-B/C (4,000 draws). Save prompts' hashes, source hashes, raw continuations,
parsed results, model ids, software versions, job ids, and runtime. Parse failures remain
in the denominators. The model never receives the correct test calculation in R7-B.
Report multiplicity and the limited three-model scope; distinguish the registered
R7-A sign rule from exploratory factorial contrasts. Allocate at most two concurrent
single-GPU workers on a30/h100; do not interfere with other projects' queued jobs.
