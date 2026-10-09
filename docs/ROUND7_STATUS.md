# Reviewer follow-up checklist

Updated 2026-10-09. Protocol: [ROUND7_PROTOCOL.md](ROUND7_PROTOCOL.md).

- [x] Freeze the response-control design and select independent computations.
- [x] R7-A: paired CoT-minus-answer-only analysis of all 44 existing cells.
  Ten reductions pass the three-demonstration-set sign rule and 95% interval criterion;
  19 intervals are below zero. Answer-only already meets equivalence in 31 cells;
  CoT meets it in 43. Source: `results/summary/round7_paired_prompting.json`.
- [x] CPU unit checks for conservative calculation eligibility and parsing: seven passed.
- [x] Tokenizer preflight on a compute node: job 450293 completed successfully.
  All four full prompts have equal token lengths across formats and matched names,
  for every selected problem, model, demonstration seed and candidate example count.
- [ ] R7-B: format factorial and independent neutral competence checks.
- [ ] R7-C: fixed correct generated calculations, crossed final-query and output formats.
- [ ] Independently validate the GPU outputs and collect response-control results.
- [ ] Incorporate completed evidence, rebuild the PDF/source bundle, verify and push.

Two single-GPU workers were initially submitted as 450291 (a30, small models sequentially)
and 450292 (h100, Llama-3.1-8B), with the successful CPU preflight as their dependency.
Both initially waited for resources/priority; they were not running when submitted.

**Reason for allowing an h200 fallback:** all a30 GPUs are occupied, the small-model job's
initial estimated start was October 10, and the h100 job had no estimated start. ARR is
October 12. These workers need inference only and at most two GPUs concurrently. Moving
the same pending jobs to h200 avoids duplicate work; SLURM enforces the account's shared
QoS limits. No other project's running or queued job is cancelled or modified.
Actual routing and states are recorded in `log/round7_2026-10-09/`.

The final-query cohort is conditional on completely correct original generations. Both
Llama models supply 200 unique computations per seed/target. OLMo supplies 144/108/116 for
the queried target and 44/8/25 for the intermediate target; these smaller samples remain
visible. Repeating the queried name repeats the misleading name only in queried-target
conditions; intermediate-target conditions provide a control for query repetition.
