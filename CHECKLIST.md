# Experiment checklist

One line per experiment the paper needs. Status is the truth on disk, not a plan: an item is
ticked only when its numbers are in `results/summary/` and reach the paper through
`scripts/51_tables.py`. Details and numbers: `docs/EXPERIMENTS.md`, `results/NOTES.md`.

## Done

- [x] Tokenizer rule-4 check and verified word lists (E0)
- [x] Kudo et al. replication on letters, Llama-3.2-3B (E2, E27–E30: t*_eq reproduced)
- [x] Behaviour, level 3, eight models, three demonstration sets (E3, E14–E16, E33)
- [x] Behaviour, levels 1–5, Llama-3.2-3B and Llama-3.1-8B, three demonstration sets (E13, E33)
- [x] Neutral-trained probes with selectivity control: Llama-3.2-3B (L2–L4), Llama-3.1-8B, Gemma-3-4B, OLMo-2-7B, OLMo-2-1B (E5, E6)
- [x] Patching with both controls, removal and injection, four models (E8–E10, E35)
- [x] No-computation control, level 1 (E34)
- [x] Irrelevant-lure control, level 4, against the matched baseline (E12, `63_irrelevant.py`)
- [x] Free-form chain on instruct models (E36)
- [x] Equivalence bounds, all five levels (E37)
- [x] Value-written control (`62_value_written.py`)
- [x] Value-step decodability vs behaviour, 8 cells (E39)
- [x] Positional-copy control, pooled (E22)
- [x] Lure-distance covariate (E19), teacher-forced lure rate (E20), error taxonomy (E26)
- [x] Identifier-name cue, level 3, four models, three demonstration sets (E38)

## To do before submission (2026-10-12)

**No GPU experiments remain.** Everything below is CPU analysis, writing or submission admin.

- [ ] Regenerate every figure after the final data (`52_figs`, `53_kudo_figs`, `58_token_figure`, `60_master_figure`)
- [ ] `99_selfcheck.py` with zero failures on the final state
- [ ] Final `make paper`: body ≤ 4 pages, no red `\NUM`, no overfull boxes
- [ ] Anonymise for review (author block, repository link, acknowledgements)
- [ ] ARR checklist and Limitations section reviewed against the final text

## Not planned for this paper

E7 (instance-level margin link), E11, E17 (probe recipe robustness), E18 (patching scope),
E21 (mixed-effects), E23 (lure index), E24 (already reported as normalised LD), E31 (Simple CoT),
E32 (operand-copy error at level 2).
