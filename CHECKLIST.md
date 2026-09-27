# Experiment checklist

One line per experiment the paper needs. Status is the truth on disk, not a plan: an item is
ticked only when its numbers are in `results/summary/` and reach the paper through
`scripts/51_tables.py`. Details and numbers: `docs/EXPERIMENTS.md`, `results/NOTES.md`.

*Last updated 2026-09-26. ARR deadline 2026-10-12 (AoE).*

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

- [ ] **G1. Chain-of-thought patching of the exception model (recommended).** OLMo-2-1B-Instruct is the
      only model whose CoT answers follow the name (110–131 lure errors per demonstration set), but its
      CoT patching ran only `inject ctl_word`; the neutral-twin patch that removes lure errors and the
      alternative-lure control were never run. They test directly whether its CoT lure errors come from
      the name's early-layer representation, as the no-CoT errors do in Llama-3.2-3B. All caches exist.
      `python scripts/40_patch.py --model olmo2-1b-it --level 3 --regime cot --contrasts main ctl_lure --limit 500`
      (a30 or h100, under 1 GPU-hour). Optional follow-on: the span grid,
      `--grid --sources neutral incongruent_alt --limit 500`.
- [ ] **G2. Probes for three more panel models (optional).** The claim that the effect tracks whether
      the value is represented rests on 10 cells from 5 models. Gemma-3-12B-it, Llama-3.1-8B-Instruct and
      Llama-3.2-3B-Instruct have behaviour runs only (`--no-train --no-forced`). A full
      `pipeline.py --stage all` for them adds 6 cells. Roughly 2–4 GPU-hours each; the 12B needs h100/h200.
- [ ] **G3. Per-token probes on a non-Llama model (optional).** Llama-3.1-8B makes the queried value
      decodable at the restatement step, two equations before the 3B does. E27+E28 on Gemma-3-4B-it would
      show whether that generalises. About 2 GPU-hours.
- [ ] G4. Instruction-tuned models with their chat template. Not for this paper: it changes token
      positions for the whole design. Stays a Limitation.

### CPU analyses

- [ ] **C1. Equivalence at 90% intervals.** Amendment 3 fixes the 90% CI (two one-sided 5% tests);
      `57_equivalence.py` uses the 95% CI (`ci90()` is defined but unused). Switch and regenerate, or
      disclose the conservative deviation in the appendix.
- [ ] **C2. Equivalence for accuracy contrasts under CoT.** The body now says 15 accuracy contrasts in
      5 models survive with CoT; only the lure excess is bounded. Run the same TOST on facilitation and
      interference so that sentence has a margin behind it.
- [ ] **C3. Written-value control for the exception, pooled.** The OLMo excess quoted in the body
      (`vw-olmo-*`) is demonstration set 7 only. Pool the three sets, as everywhere else. Also check the
      OLMo L3 cot v2 row of `value_written.json`, whose `excess_not` lies outside its own interval.
- [ ] **C4. Copy-control coverage.** The pooled positional-copy control omits the four ladder checkpoints
      and Gemma-3-4B sets 11 and 13 (no `copy_control.json`). Run `50_analysis.py` on those regimes.
- [ ] C5. After any item above: `50_analysis` → `51_tables` → `60_master_figure` → `66_figure_dump` →
      `99_selfcheck` → `make paper` (body ≤ 4 pages, no red `\NUM`, no overfull boxes).

### Author decisions

- [ ] **D1. OLMo-2-1B in Table 1.** Preregistration §6 keeps models below 90% neutral CoT accuracy at
      level 3 out of Table 1; OLMo-2-1B is shown at 47% and no amendment covers it. Move it to the appendix
      (frees about three body lines) or add a dated amendment saying why it is shown.
- [ ] D2. 90% vs 95% (C1): switch or disclose.
- [ ] **D3. Read the rescoped abstract, introduction and conclusion** (commits `3aa13a0`, `17c4eb2`,
      `43ff5be`): "eleven of the models", lure answers within the margin, probes read the lure just before
      the name is written, patching scoped to Llama-3.2-3B.

### Submission

- [ ] ARR responsible-NLP checklist: answers drafted in `docs/ARR_CHECKLIST.md`; confirm the risks (A2) and
      AI-assistant (E1) wording, check against the live form, submit
- [ ] Every author registered as an ARR reviewer by 2026-10-12 (PLAN.md)
- [ ] Final build after the last change; check the PDF uploaded is `paper/main.pdf` at HEAD
- [x] Anonymised: author block, no repository link, no acknowledgements, PDF metadata, no identifying
      `note` in the references (2026-09-25/26)
- [x] Bibliography: all 24 cited entries verified against arXiv / ACL Anthology / NeurIPS / Crossref,
      published versions preferred (2026-09-26)
- [x] Claims audit of body and appendix against `results/summary/` and the raw outputs (2026-09-25/26)
- [x] Limitations re-read against the final results (2026-09-25)

## Not planned for this paper

E7 (instance-level margin link: fitted and reported as unresolved in the appendix), E11,
E24 (already reported as normalised LD), E32 (b)–(c) (operand-copy error; superseded by E33).
