# Reviewer follow-up checklist

Updated 2026-10-09. Protocol: [ROUND7_PROTOCOL.md](ROUND7_PROTOCOL.md).

- [x] Freeze the response-control design and select independent computations.
- [x] R7-A: paired CoT-minus-answer-only analysis of all 44 existing cells.
  Ten reductions pass the three-demonstration-set sign rule and 95% interval criterion;
  19 intervals are below zero. Answer-only already meets equivalence in 31 cells;
  CoT meets it in 43. Source: `results/summary/round7_paired_prompting.json`.
- [x] CPU checks for calculation eligibility, parsing, paired subtraction and competence
  gates: nine new tests passed; the full suite has 75 passes and six tokenizer skips.
- [x] Update the paper with the completed paired evidence and the distinction between
  reliable reductions and small effects; retain future-work labels for pending GPU tests.
- [x] Tokenizer preflight on a compute node: job 450293 completed successfully.
  All four full prompts have equal token lengths across formats and matched names,
  for every selected problem, model, demonstration seed and candidate example count.
- [ ] R7-B: format factorial and independent neutral competence checks.
- [ ] R7-C: fixed correct generated calculations, crossed final-query and output formats.
- [ ] Independently validate the GPU outputs and collect response-control results.
- [ ] Incorporate completed GPU evidence, rebuild the PDF/source bundle, verify and push.

Two single-GPU workers were initially submitted as 450291 (a30, small models sequentially)
and 450292 (h100, Llama-3.1-8B), with the successful CPU preflight as their dependency.
Both initially waited for resources/priority; they were not running when submitted.

**Reason for allowing an h200 fallback:** all a30 GPUs are occupied, the small-model job's
initial estimated start was October 10, and the h100 job had no estimated start. ARR is
October 12. These workers need inference only and at most two GPUs concurrently. Moving
the same pending jobs to h200 avoids duplicate work; SLURM enforces the account's shared
QoS limits. No other project's running or queued job is cancelled or modified.
Actual routing and states are recorded in `log/round7_2026-10-09/`.

The pending jobs were replaced by **450333** (small models sequentially) and **450334**
(Llama-3.1-8B), both h200 with explicit juno-pri QoS, one GPU and a four-hour limit each.
The old pending jobs 450291/450292 were cancelled before either ran. The successful
preflight has already finished; its checks are retained in the audit rather than depending
on a completed job removed from SLURM's live job table. A CPU collector depends on both
current GPU jobs and will independently check parsing, cohort identity and fixed calculations.
The arithmetic-computation clustering sensitivity also retains all 10 reliable reductions.
Collector job: **450341** (CPU only, after successful completion of both current workers).
The current routing allows 450333 on a30/h100/h200 and 450334 on h100/h200, with the
slow MIG node g-06-01 excluded. This lets SLURM use the earliest eligible allocation;
each worker still requests just one GPU. Results remain pending.

The final-query cohort is conditional on completely correct original generations. Both
Llama models supply 200 unique computations per seed/target. OLMo supplies 144/108/116 for
the queried target and 44/8/25 for the intermediate target; these smaller samples remain
visible. Repeating the queried name repeats the misleading name only in queried-target
conditions; intermediate-target conditions provide a control for query repetition.
