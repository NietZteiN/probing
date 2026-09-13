# The argument: what the paper claims, in what order, and what each claim rests on

*Written 2026-09-12, before any model has been run. This is the spine of the short paper.
PLAN.md is the science; docs/EXPERIMENTS.md is what runs; this file is how the paper is put
together. Change it when the argument changes, not when a number changes.*

## 0. Title and thesis

**Title (default):** *Name or Value? Cue Conflict Reveals When Chain of Thought Overrides
Lexical Priors.* Names the method and the finding space, which is what reviewers scan for.
Fallback if the finding is negative for override: *When the Name Lies: Probing Lexical
Interference During Chain-of-Thought Arithmetic.*

**Thesis (one sentence, every section serves it):** When a variable's name contradicts its
value, does chain of thought let the model's computation override the name, and where inside
the model does that happen?

## 1. The claim ladder

Each claim rests on the one before it. A reader who stops after any rung has learned
something complete.

| # | claim | evidence | figure/table | script |
|---|---|---|---|---|
| C1 | Without a chain, models read a variable's value off its name: a congruent name raises accuracy (up to +20 points) and an incongruent one produces lure answers (up to +51 points above baseline where accuracy is at floor) | facilitation and lure excess, direct regime, three demonstration sets, claim rule of amendment 2 | Table 1 | `56_seed_sweep`, `51_tables` |
| C2 | With a chain, the name's value has no effect on the answer at any level or model: lure excess within ±0.5 points, congruent and incongruent move together | same quantities, chain regime | Table 1 | same |
| C3 | Inside the model the lure is read only where the name token sits (and at the query, where the readout is name identity); the true value appears at the value step in every condition and the lure is erased there (lure mass .004); an unbound number word's value lingers (.118) | per-token and labelled-position probes with selectivity | Figure 2 | `30_train_probes`, `53_kudo_figs` |
| C4 | Without a chain the lure answer is tied to the early-layer representation of the span where the name is defined (neutral-twin patch removes it, alternative lure is not followed); the word control's damage bounds how much of that is disruption | Figure 3 with both controls | Figure 3 | `40_patch`, `54_grid_figs` |
| C5 | What a chain does not remove is a word-class effect on equation selection: a number-word name on the queried variable at level 4 costs the 3B model 10 points whether or not its value agrees | interference = −facilitation, both claimable | Table 1 / text | `56_seed_sweep`, E26 |
| C6 | Every one of these is smaller than the effect of which three demonstrations are shown; the demonstration set is a reported factor | across-seed ranges | Appendix | `56_seed_sweep` |

C1 is the paper's floor: if there is no interference on either core model, there is no paper
in this cycle (PREREGISTRATION §6 and EXPERIMENT_PLAN §3). C3 is the ceiling: it is the part
no behavioural study can provide and the reason to cite this paper.

## 2. The outcome table, pre-written

The introduction and abstract are drafted before results because every cell is publishable.
The sentence to paste is written here so the fill-in on result day is mechanical.

| lure inside the model | model answers correctly | model answers with the lure |
|---|---|---|
| **fades** (margin crosses at K) | "Computation overrides the name: the lure is decodable through step K and then replaced, and the answer is correct. This extends Kudo et al.'s claim to conflicting input." | "The error enters late: the true value is present at step K but the readout at P5 follows the name. A new failure location, at output rather than in computation." |
| **persists** (margin never crosses) | "Hidden interference: the output is correct but the internal state carries the lure to the end. Correct behaviour is not evidence of a clean representation." | "Faithful computation from a corrupted value: the chain writes the lure and the answer follows it. The chain of thought is faithful and wrong." |

The regime contrast then says which cell the model moves *between* when it writes the chain.
**Headline after the runs (2026-09-13):** *without a chain, models read a variable's value off
its name; with a chain, the name's value is irrelevant to the answer, and the probes show the
lure is read only where the name sits and is erased at the step that computes the value.* The
outcome cell is "lure fades, answer correct", reached not because the chain suppresses the
lure but because writing the value replaces it. The residue is a word-class effect on
equation selection, and the largest effect of all is the demonstration set.

## 3. Section by section (4 pages of body)

| section | budget | says | must contain |
|---|---|---|---|
| Introduction | 0.75 | names interfere (code); drops cannot localise; clean-task probes cannot tell computed from read; cue conflict can; findings + 3 contributions | Figure 1 (three twins + margin sketch); the outcome-cell sentence |
| Setup | 0.75 | task, conditions, generation rules, regimes, positions P1–P5 (P1 excluded), probes trained on neutral only, patching with two controls, models, metrics | one running example reused everywhere |
| Results | 1.75 | replication line; RQ1 (Table 1); RQ2 (Figure 2 + crossover + instance link); RQ3 (Figure 3 + controls) | one paragraph per RQ, each ending with the cell it lands in |
| Related work | 0.5 | the six-row table in prose | Kudo first, as the paper built on |
| Conclusion | 0.25 | the finding; one implication for faithfulness | nothing new |
| Limitations | free | synthetic, ≤8B, English number words, linear probes, format writes name beside value | required by ARR |
| Appendix | free | tokenizer checks, full sweeps, selectivity, extra models and levels, mixed-effects fit | everything a reviewer asks for in §5 |

Figure 1 must carry the idea alone: a reader who sees only it should understand the design.
Figure 2 is the ceiling claim; it must show CoT and direct side by side on the same colour
scale. Figure 3 must show all three curves; recovery without the controls is not a result.

### Appendix figures that mirror Kudo et al.

Reviewers who know Kudo et al. will look for their figures. The appendix carries their Fig. 2
(per-token accuracy heatmap + max-over-layers curve) for neutral, congruent and incongruent
side by side, with lure-rate and margin heatmaps under the incongruent column; their Tables 2–3
per condition with t_lure added; their Figs. 5–6 patching grid with matched-twin sources; and
their Fig. 3 trajectories on lure-error instances (docs/EXPERIMENTS.md §F). Figure 2 of the
main text is the incongruent column of that figure, chain beside direct.

## 4. What breaks each claim, and the fallback

| claim | breaks if | fallback |
|---|---|---|
| C1 | no interference on either core model | negative result is not a short paper; move to next cycle, add 8B-instruct and a harder level |
| C1 | interference exists but errors are not the lure | report "generic disruption" honestly; the probe story (C3) becomes the paper |
| C2 | both regimes equal | fine: the contrast is still the centre, the answer is "CoT does not help" |
| C3 | probes below selectivity threshold at P2–P5 | report the cells that pass; rely on C4; state that the value is not linearly readable there |
| C3 | crossover differs by layer | report layer-max and the appendix sweep; do not pick a layer post hoc |
| C4 | control patch damages > 5% | narrow to prompt-only positions (`--scope prompt`); if still damaged, drop the causal claim and keep C1–C3 |
| C4 | recovery ≈ 0 at every layer | the name's influence is distributed; say so; the ctl_lure result still shows the name matters |

## 5. Reviewer questions and where each is answered

| question | answered in |
|---|---|
| Is this just a robustness check of Kudo et al.? | Intro ¶2 and Related work: cue conflict tests what the decoded signal reflects; clean tasks cannot |
| Won't a probe trivially decode "two" as 2? | Setup, positions: P1 excluded; probes trained on neutral only; claims at P2–P5 |
| Why arithmetic, not code? | Setup and Limitations: matched twins are impossible on real code at scale |
| Why these names, not optimised attacks? | Limitations: the question is whether meaningful names interfere; "misleading", never "adversarial" |
| Probing is correlational | RQ3 with two controls |
| Could tokenisation explain it? | Appendix A: per-model word lists, one token in every context, identical layouts across twins |
| Only small models | Limitations; within-family size trend |
| Mixed effects? | Appendix: random intercept per matched set beside the clustered logit |

## 6. Writing timeline

| date | writing deliverable |
|---|---|
| Sep 17 | replication paragraph; Figure 1 drawn from a real instance |
| Sep 24 | Table 1 filled for Llama-3.2-3B; RQ1 paragraph |
| Oct 1 | Figure 2 and 3 for both core models; RQ2/RQ3 paragraphs; **go/no-go** |
| Oct 5 | abstract and intro filled from the outcome cell; related work trimmed to 0.5 page |
| Oct 8 | full draft to co-authors; limitations; appendix |
| Oct 10 | anonymisation check, responsible-NLP checklist, reviewer registration for every author |
| Oct 12 | submit (AoE) |

## 7. Conventions

Misleading / incongruent, never adversarial. Interference and facilitation in the Stroop
sense. P1 is a reference, never evidence. No number typed by hand: `\NUM{}` until
`51_tables.py` fills it. Claims stay on arithmetic with variable assignment.
