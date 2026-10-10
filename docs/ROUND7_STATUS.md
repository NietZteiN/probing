# Reviewer follow-up checklist

Updated 2026-10-10. Protocol: [ROUND7_PROTOCOL.md](ROUND7_PROTOCOL.md).

- [x] Freeze the response-control design and select independent computations.
- [x] R7-A: paired CoT-minus-answer-only analysis of all 44 existing cells.
  Ten reductions pass the three-demonstration-set sign rule and 95% interval criterion;
  19 intervals are below zero. Answer-only already meets equivalence in 31 cells;
  CoT meets it in 43. Source: `results/summary/round7_paired_prompting.json`.
- [x] CPU checks for calculation eligibility, parsing, paired subtraction and competence
  gates, including rejection of duplicate pairs, changed arithmetic clusters and wrong
  decoding budgets. Reporting checks retain parse failures and failed competence checks.
  Full suite: 79 passed, six tokenizer-dependent tests skipped; 13 follow-up tests passed.
- [x] Update the paper with the completed paired evidence and the distinction between
  reliable reductions and small effects; incorporate the completed response controls below.
- [x] Tokenizer preflight on a compute node: job 450293 completed successfully.
  All four full prompts have equal token lengths across formats and matched names,
  for every selected problem, model, demonstration seed and candidate example count.
- [x] Prepare automatic validated CSV summaries and optional appendix tables. The collector
  records source hashes, event counts, accuracy, parse coverage and cohort sizes by seed.
  Pending runs produce no manuscript tables; the completed, validated results are now included.
- [x] R7-B: format factorial and independent neutral competence checks.
- [x] R7-C: fixed correct generated calculations, crossed final-query and output formats.
- [x] Independently validate the GPU outputs and collect response-control results.
- [x] Incorporate completed GPU evidence, rebuild the PDF/source bundle, verify and push.

## Queue history

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
on a completed job removed from SLURM's live job table. The CPU collector depended on both
GPU jobs and independently checked parsing, cohort identity and fixed calculations.
The arithmetic-computation clustering sensitivity also retains all 10 reliable reductions.
Collector job: **450341** (CPU only, after successful completion of both workers).
The routing allowed 450333 on a30/h100/h200 and 450334 on h100/h200, with the
slow MIG node g-06-01 excluded. This lets SLURM use the earliest eligible allocation;
each worker still requests just one GPU. Both GPU workers and the collector completed successfully.

Queue checked again October 9 at 16:46 Central: both GPU workers remained pending for
priority, with no calibration or experiment outputs. The scheduler estimates October 10
at 14:08 for 450333 and gives no estimate for 450334; estimates can change. The collector
was waiting for successful completion of both. Historical audit: `continuation_status.json`.

The final-query cohort is conditional on completely correct original generations. Both
Llama models supply 200 unique computations per seed/target. OLMo supplies 144/108/116 for
the queried target and 44/8/25 for the intermediate target; these smaller samples remain
visible. Repeating the queried name repeats the misleading name only in queried-target
conditions; intermediate-target conditions provide a control for query repetition.

## Completed results

Completion verified October 10: jobs 450333, 450334 and 450341 exited successfully.
All raw outputs were independently reparsed and checked against frozen computations and
correct-calculation cohorts. Validated results: `results/summary/round7_response_controls.json`.

- Named and unnamed numeric equations both reach 99--100% ordinary-name accuracy in the
  Llama models. These four role comparisons meet the held-out competence criterion.
- No model passes the common calibration criterion across all four formats; all use
  16 examples. Values-only comparisons do not match competence, and are reported accordingly.
- Neither Llama returns a name-suggested digit after any fixed correct calculation.
  This is a finite-cohort observation, not a zero population-rate claim.
- OLMo's unnamed-query/digit-output condition has 12.8 points of added errors [9.1, 16.6]
  over 368 paired runs (205 distinct computations). Repeating the queried name lowers
  this to 0.5 points, while ordinary-name accuracy changes from 78.0% to 96.2%.
  No OLMo final-format comparison meets the competence criterion.
- Paper inclusion uses a concise implications paragraph and two appendix tables. The
  complete contrasts remain in validated JSON/CSV artifacts rather than a table gallery.
