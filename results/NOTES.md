# Results log

*One entry per result that changes what the paper can say. Numbers here are copied from
`results/summary/`; the paper reads them from `paper/numbers.tex`, never from this file.*

## 2026-09-12 — Level 3, Llama-3.2-3B (full) and Llama-3.1-8B (behaviour only)

**Behaviour (Table 1).** 2,000 matched sets per condition, cluster-bootstrap 95% CIs.

| model | regime | target | neutral | congruent | incongruent | lure rate | pseudo-lure |
|---|---|---|---|---|---|---|---|
| 3B | CoT | queried v1 | 96.6 | 95.1 | 94.6 | 0.4 | 0.1 |
| 3B | CoT | interm. v2 | 96.6 | 99.5 | 99.4 | 0.0 | 0.1 |
| 3B | direct | v1 | 13.4 | 18.9 | 13.9 | 8.6 | 4.7 |
| 3B | direct | v2 | 13.4 | 17.9 | 13.8 | 8.1 | 4.7 |
| 8B | CoT | v1 | 85.7 | 82.5 | 83.6 | 0.7 | 1.1 |
| 8B | CoT | v2 | 85.7 | 97.5 | 84.0 | 0.9 | 0.9 |
| 8B | direct | v1 | 41.3 | 48.8 | 44.4 | 3.5 | 3.1 |
| 8B | direct | v2 | 41.3 | 47.9 | 45.8 | 3.5 | 3.5 |

- With a chain, the lure is essentially absent as an answer (≤ 1%, at the pseudo-lure baseline
  for 8B). Interference on the queried variable is about 2 points for both models and is made of
  non-lure errors.
- Facilitation is the large effect: a congruent name on the intermediate gives +2.9 (3B) and
  +11.8 (8B) points; in the direct regime a congruent name gives +5 to +7.5.
- Without a chain the 3B model is at floor (13%, answers `name=0` 44% of the time) but the lure
  share doubles over baseline (8.6 vs 4.7); the 8B model is at 41–49% with the lure share AT
  baseline (3.5 vs 3.1–3.5): no behavioural interference at all.
- A number-word name helps even when incongruent for 3B-CoT-v2 (99.4 vs 96.6) and for 8B-direct
  (44–46 vs 41). Registered as E26 (word-class effect vs value effect).
- Letters: 3B 98.7 (Kudo: 93.2), 8B 96.6 (Kudo: 99.4).

**Replication (E2, 3B, letter-trained probes, CoT).** First position above τ = 0.9 is the chain
step where the value is written, for both variables (Kudo: t*_eq > 0); pre-CoT maximum 0.17 (v1)
and 0.48 (v2) (Kudo: 17.8 / 33.2 for Llama-3.2-3B). Reproduced.

**Representation (E5/E6, 3B, neutral-trained probes, incongruent).** At the layer with the best
neutral accuracy per position:

| regime | target | P2 end of definition | P3 query | P4 value step | P5 answer |
|---|---|---|---|---|---|
| CoT | v2 | acc .48, lure .07, margin +1.5 (sel. .32) | acc .16, **lure .32**, margin −0.4 (**sel. fails: ctl .97**) | acc 1.00, lure .00, margin +6.2 | acc 1.00, lure .00, +4.6 |
| CoT | v1 | acc .17, lure .11, +0.2 (sel. fails) | acc .07, **lure .62**, −1.7 (**ctl 1.00**) | acc 1.00, lure .00, +6.8 | same |
| direct | v2 | acc .46, lure .06, +1.5 | acc .24, **lure .43**, −0.3 (ctl .96) | — | acc .69, lure .02, +4.9 (sel. .49) |
| direct | v1 | acc .17, lure .11 | acc .06, **lure .66**, −1.8 (ctl 1.00) | — | acc .38, lure .07, +1.5 (**ctl .91**) |

- The lure dominates the readout at the query position, but the control task also scores
  0.96–1.00 there: at P3 (last layer) the probe reads the name's identity, not a computed value.
  Under the pre-registered selectivity threshold that cell is reported as "name identity
  present", not as "lure value represented".
- At P2 the true value is partly computed already (acc ≈ .48 for the intermediate, selectivity
  .32) and the lure is weak (.06–.07).
- At P4/P5 with a chain the lure is gone (0%) and the true value is fully decodable. Outcome cell
  for 3B-CoT: **lure fades, answer correct**.
- In the direct regime the intermediate's true value is still decodable at the answer position
  (.69, lure .02); the queried variable's probe there fails selectivity.

**Causation (E8, 3B, main contrast only; controls pending).** CoT: 1 and 0 lure errors, nothing
to recover; normalised LD ≈ 1.0. Direct: 172 / 152 lure errors; patching the name positions
from the neutral twin removes the lure answer in 52% (v1) / 39% (v2) at all layers and in 75% /
46% at the best single layer, **layer 1** (also 51% / 43% at layer 0): the lure answer is caused
by the name token's earliest-layer representation. Normalised LD ≈ 1.0 (margin fully restored).
"Recovery to the correct answer" is 1–2% because the neutral twin itself is right only 13% of
the time; the metric for this regime is lure removal (and matches-source, recorded from now on).

**Open.** 8B probes and patching (queued); ctl_word / ctl_lure for 3B (queued); E22 has 9 lure
errors to look at; E26 error analysis.

## 2026-09-12 (later) — control patches, Llama-3.2-3B, direct regime

| target | layer set | neutral patch: lure removed | normalised LD | word control: damage | lure control: follows new lure | lure control: lure removed |
|---|---|---|---|---|---|---|
| v1 | L1 | .75 | .83 | **.43** | .09 | .37 |
| v1 | W0-3 | .53 | 1.00 | .21 | .09 | .36 |
| v1 | ALL | .52 | 1.00 | .22 | .09 | .35 |
| v2 | L1 | .46 | .82 | .21 | .07 | .22 |
| v2 | ALL | .40 | .99 | .17 | .07 | .18 |

- **The word control fails the pre-registered 5% damage threshold** (17–43%): at 13% baseline
  accuracy, replacing the name token's activations with another neutral word's changes a fifth
  of the few correct answers. Patching in this regime is disruptive, so "lure removed" is not by
  itself evidence that the name causes the lure answer.
- What survives: the neutral patch removes the lure answer more often than the alternative-lure
  patch does (52% vs 35% for v1, 40% vs 18% for v2, all layers) and fully restores the
  true-minus-lure margin (normalised LD ≈ 1.0), and swapping one lure for another makes the
  answer follow the new lure only 7–9% of the time, near the 4.7% base rate. Reading: the
  lure answer is tied to the name token's representation, but the effect is not cleanly
  separable from disruption at floor accuracy. Per PREREGISTRATION §5 the causal claim for
  3B-direct is downgraded; the 8B model (41–49% direct accuracy) decides whether a clean version
  exists.
- In the chain regime both controls are clean (damage 0.1%, follows-new-lure 0%), but there are
  no lure errors to recover; the chain's value step is where the name stops mattering.
- Job hygiene: the two 8B chain-regime probe jobs hit their 2 h limit before the second variable
  and the manifests recorded exit 0 (SLURM's SIGTERM bypassed the EXIT trap). Fixed in
  `submit.py` (TERM now records 143); the missing probes are resubmitted with 3 h.

## 2026-09-12 (later) — patching with controls, Llama-3.1-8B

Direct regime (41–49% accuracy; behavioural lure rate 3.5% vs pseudo-lure 3.1–3.5%, i.e. no
lure effect above chance):

| target | layer set | lure errors | neutral patch: lure removed | matches neutral twin's answer | normalised LD | word control: damage | lure control: follows new lure | lure control: lure removed |
|---|---|---|---|---|---|---|---|---|
| v1 | ALL | 80 | .43 | .97 | 1.00 | .10 | .04 | .39 |
| v1 | L1 | 80 | .48 | .80 | .46 | .21 | .03 | .43 |
| v2 | ALL | 72 | .31 | .95 | 1.01 | .09 | .04 | .31 |

- The word control is at 9–10% damage (all layers), still above the 5% threshold but far below
  the 3B's 17–22%. Patching the name positions from the neutral twin makes the incongruent run
  produce the neutral twin's answer 95–97% of the time.
- **The alternative-lure patch removes the "lure" answer exactly as often as the neutral patch
  (39–43% vs 43%, 31% vs 31%) and the answer follows the new lure only 4% of the time, at the
  base rate.** Together with the behavioural lure rate being at chance, this says the 8B
  direct-regime "lure errors" are coincidences (a wrong answer that happens to equal the lure),
  not name-driven, and the patching outcome is the disruption rate of any name swap. There is
  no name-caused lure effect to localise in the 8B model.
- Chain regime: zero lure errors; both controls at 0.

**Emerging headline across both core models.** Number-word names barely mislead these base
models: with a chain of thought the lure is absent from behaviour (≤ 1%) and from the
representation by the value step; without a chain only the 3B model shows a lure effect above
chance (8.6% vs 4.7%), and its patching evidence is confounded by disruption at floor accuracy.
The large, robust effect is **facilitation**: a congruent name is used when it agrees with the
computation (+3 to +12 points), and a number-word name changes accuracy even when its value is
irrelevant (E26). The paper's outcome cell is "lure fades, answer correct", with the twist that
the name's value is *recruited* rather than *overridden* when it helps.

## 2026-09-12 (later) — Llama-3.1-8B probes complete

Same picture as the 3B, chain regime, neutral-trained probes, incongruent, best-neutral-accuracy
layer: end of definition acc .45 / lure .06 / margin +1.6 (v2); query lure .40 (v2) and .55 (v1)
with control accuracy 1.00 (name identity, not claimed); value step and answer acc .99–1.00,
lure .000, margin +4.4 to +6.3. Replication: first position above τ is the value step for both
variables; pre-chain maxima .23 (v1) / .53 (v2) vs Kudo's 26.0 / 29.6 for Llama-3.1-8B.
Kudo Fig. 5 reproduced in our format (different-problem source): the final answer flips only
when the substitution step (layers 0–15) or the value step (layers 16–27) is patched; no input
equation or earlier step has any effect (paper/figures/kudo_fig5_llama32-3b_L3_cot_v2.png).

## 2026-09-12 (later) — E19, E20, E26 on both core models, level 3

**E26, what the chain errors are.** Llama-3.1-8B: 85–95% of all wrong chains in every
condition are a single error at the intermediate's value step: it writes the FIRST equation's
digit operand as the intermediate's value (`truck=2 + ant, ant=7 - 6, ant=2`; `cloud=9 + pear,
pear=0 - 0, pear=9`), then carries it through. Counts: neutral 245/287, incongruent@v2 291/321,
congruent@v2 47/50, incongruent@v1 300/328. A congruent number-word name on the intermediate
removes most of these errors (287 → 50 wrong); an incongruent one neither adds lure errors
(3 and 9 "wrote the lure" out of 2,000) nor removes the copy errors. Llama-3.2-3B's errors are
mixed (wrong expression, substitution of a wrong operand at step 4, other values) and fewer.
**Reading:** the dominant failure inside the chain is a positional copy of a salient earlier
number (the in-chain analogue of Liu 2026's readout shortcut), not the name's value. A number
word that agrees with the value acts as a correct cue against that copy; a number word that
disagrees is simply not used. The lure never wins.
**E20.** With the gold chain forced up to the value step, the model writes the lure there
0.0–0.15% of the time (both models); free-generation lure rates are 0.4–0.9%. **E19.** No
monotone effect of |lure − true| on lure rate; accuracy slopes are small and of inconsistent sign.

## 2026-09-13 — E29 grids, Llama-3.2-3B (500 pairs per cell, 4-layer windows)

- **Different-problem source (Kudo's design), chain regime:** the final answer follows the
  source only when the substitution step (layers 0–15, up to 77%) or the final value step
  (layers 16–27, 100%) is patched; input equations, the query and earlier steps do nothing.
  Kudo et al.'s recency-bias grid reproduced in our format. In the direct regime no single span
  moves the answer above 18% (the model is at 13% accuracy; the answer is not localised).
- **Neutral-twin source, direct regime, lure on the queried variable:** the lure answer is
  removed most by patching the queried variable's own equation span at layers 4–7 (68%), then
  the query span at layers 0–3 (46%) and the answer step (40%); patching the other equation
  does nothing. The alternative-lure source moves the answer to the new lure in ≤ 10% of cases
  anywhere. So the direct-mode lure answer is tied to the early-layer representation of the
  span where the misleading name is defined, not to the name token alone.
- **Chain regime:** 0–1 lure errors in 500, so lure-removal cells are undefined (masked in the
  figure when the denominator is below 20).

## 2026-09-13 — E28/E30 per-token probes, Llama-3.2-3B, chain regime, intermediate variable

Kudo Fig. 2 analogue (paper/figures/kudo_fig2_llama32-3b_L3_cot_v2.png): the value becomes
decodable (τ = 0.9) at token 14 after the chain starts, the `=` of the intermediate's value
step, identically in neutral, congruent and incongruent; pre-chain maxima .46 / .47 / .43 (letters
.50). The lure rate (max over layers) peaks only at the tokens where the number-word name itself
occurs (up to .79 on a name token in the chain, .31 in the input) and is near zero at every value
position. The control-task heatmap shows name identity linearly recoverable in the last ~4 layers
at most tokens, which is exactly where the query-position "lure" reads come from. Trajectories
(Fig. 3 analogue): on random incongruent instances the top-1 probe prediction is the lure at
name tokens and the true value from the value step onward; in 2 of 12 sampled instances the
probe reads the lure across the whole input, worth a look (E30 follow-up).

## 2026-09-13 — E2 per-token replication, Llama-3.2-3B (letter-trained, letter test, chain)

Kudo et al. Table 3, Llama-3.2-3B row: t*_eq(A) = 5, t*_eq(B) = 2; Acc≺CoT 17.8 / 33.2.
Ours, per-token probes on the same format: first token above τ = 0.9 is the final value step
for the queried variable (segment `cot:5:value:v1`, i.e. equation 5) and the intermediate's value
step (`cot:2:value:v2`, equation 2); pre-chain maxima .30 / .48. **t*_eq reproduced exactly;
pre-chain accuracies are of the same order (ours a little higher, layer stride 2 and 4,000
training instances).**

## 2026-09-13 — Levels 2, 4, 5 behaviour, both core models (2,000 matched sets each)

| model | level | regime | neutral | letter | queried: cong / inc / lure (pseudo) | intermediate: cong / inc / lure (pseudo) | other |
|---|---|---|---|---|---|---|---|
| 3B | 2 | CoT | 94.0 | 100 | v2: 94.3 / 93.9 / 0.1 (0.1) | v1: 98.4 / 93.5 / 0.4 (0.2) | |
| 3B | 2 | direct | 29.4 | 32.0 | v2: 31.4 / 27.9 / 10.2 (6.5) | v1: 41.3 / 24.4 / 9.3 (7.5) | |
| 3B | 4 | CoT | 81.5 | 99.5 | v1: 63.4 / 62.8 / 2.4 (1.1) | v2: 80.0 / 76.5 / 1.1 (1.2) | irrelevant@v3 76.4, lure 1.8 |
| 3B | 4 | direct | 22.6 | 25.2 | v1: 22.7 / 22.6 / 0.5 (0.4) | v2: 23.1 / 23.0 / 1.1 (0.5) | irrelevant@v3 22.9, lure 0.9 |
| 3B | 5 | CoT | 63.1 | 90.5 | v1: 64.0 / 64.1 / 2.2 (2.4) | v2: 67.2 / 69.2 / 2.1 (2.5); v3: 74.2 / 72.2 / 2.2 (2.5) | |
| 3B | 5 | direct | 11.9 | 12.8 | v1: 15.8 / 11.3 / 6.8 (5.1) | v2: 12.7 / 12.0 / 6.8 (5.1); v3: 12.8 / 12.0 / 5.4 (5.2) | |
| 8B | 2 | CoT | 100 | 100 | 100 / 100 / 0 | 100 / 100 / 0 | **no operand-copy errors when the intermediate comes first** |
| 8B | 2 | direct | 37.4 | 32.0 | v2: 46.2 / 34.4 / **13.3 (6.0)** | v1: 47.4 / 36.7 / 8.5 (5.9) | |
| 8B | 4 | CoT | 100 | 100 | v1: 99.9 / 99.7 / 0 | v2: 100 / 99.9 / 0 | irrelevant@v3 100 |
| 8B | 4 | direct | 30.6 | 38.8 | v1: 46.3 / 28.6 / **7.8 (3.1)** | v2: 37.9 / 32.4 / 3.2 (3.0) | irrelevant@v3 31.6, lure 4.9 |
| 8B | 5 | CoT | 53.0 | 82.3 | v1: 49.6 / 50.2 / 4.0 (3.9) | v2: 53.1 / 56.5 / 4.0 (4.0); v3: 63.0 / 54.5 / 3.8 (3.9) | |
| 8B | 5 | direct | 21.1 | 28.7 | v1: 32.6 / 22.3 / 9.7 (4.7) | v2: 20.9 / 22.2 / 7.2 (5.3); v3: 20.2 / 21.8 / 6.2 (5.9) | |

- **The lure never wins under a chain, at any level or model** (lure rate at or below the
  pseudo-lure baseline everywhere in the CoT rows).
- **Without a chain the lure effect is real but modest and appears where accuracy is off the
  floor**: 8B level 2 queried 13.3% vs 6.0% baseline; 8B level 4 queried 7.8% vs 3.1%;
  8B level 5 queried 9.7% vs 4.7%; 3B level 2 ~9–10% vs 6.5–7.5%. Still a minority of answers.
- **Facilitation stays the large effect** (+12 to +16 points for a congruent name on the
  queried variable in direct mode; +12 for 3B at level 2 in the chain).
- **Word names are much harder than letters at depth**: 3B level 4 neutral 81.5 vs letters
  99.5, level 5 63.1 vs 90.5; 8B level 5 53.0 vs 82.3. Kudo et al.'s letter format is the easy
  case; the neutral-noun condition is a harder binding task in its own right.
- **Number-word names as a class**: at level 4 a number-word name on the QUERIED variable
  drops the 3B model from 81.5 to 63 regardless of congruence, and on the distractor
  (irrelevant control) to 76.4 with a lure rate of 1.8% (baseline ~1.1%): presence without
  binding costs accuracy but does not produce lure answers.
- **E32(a) answered**: the 8B operand-copy error disappears at level 2 (intermediate defined
  first) and at level 4, so it is tied to the level-3 layout (intermediate's equation second,
  two equations only). Check the level-3 demonstrations for a coincidence that could teach it.

## 2026-09-13 — a demonstration artefact behind the 8B level-3 copy error; E26 at levels 4 and 5

- **Level-3 demonstrations.** The three fixed word demos at level 3 include
  `fish=1 - bell, bell=8 - 7; fish=?` where the intermediate's value (1) equals the first
  equation's operand (1). One demo in three exhibits the very coincidence the 8B model then
  reproduces as its dominant error ("write the first operand as the intermediate's value").
  At level 2 and level 4 the 8B model makes essentially no errors (1–6 in 2,000), and their
  demos carry no such coincidence. Conclusion: the 8B level-3 error rate (14% neutral) is
  largely a few-shot artefact, not a property of the model; the within-condition comparisons
  stay valid because every condition shares the demos, but the absolute level-3 accuracy and
  the size of the "congruent name fixes it" effect must be re-measured with different demos.
  **E33** (registered): rerun level-3 behaviour with demonstration seed 11 (`cot_s11`,
  `direct_s11`) for both models; submitted.
- **Level 4, Llama-3.2-3B.** The dominant error (223/371 neutral; 609/732 with a number-word
  queried name) is `wrong_lhs_name` at step 2: after restating the queried equation the model
  restates the wrong equation next (it picks the distractor or the query instead of the
  intermediate's definition). A number-word name on the queried variable doubles this error
  regardless of congruence (63% vs 81.5%): equation selection, not arithmetic, is what the
  name disrupts. The 8B model makes 1–6 errors in 2,000 at level 4.
- **Level 5.** 3B: 632/720 errors are a wrong substitution at step 5 (`bell=3 + frog` →
  `bell=3 + 2` with the wrong operand). 8B: 689/1007 errors write an unrelated value at step 3,
  the deepest variable's first computation (neutral 53% vs letters 82%). Word names are
  harder than letters at depth for both models.

## 2026-09-13 — Llama-3.2-3B probes at levels 2 and 4 (neutral-trained)

- **Level 2 (intermediate defined first)** looks like level 3: value decodable at its chain
  step (acc .97–1.00, lure .000), weakly at the end of its definition (.43, lure .06), name
  identity at the query. The layout change does not change where the value appears.
- **Level 4, irrelevant-lure control (number word on the distractor v3).** The distractor's
  own value is not represented anywhere (neutral accuracy .04–.25, as Kudo et al. found for
  distractors), but the number word's digit IS readable at the query (.28, control .70) and,
  more interestingly, at the answer position in the chain regime: lure rate .15, lure mass
  .118 (control .30). Compare the bound incongruent@v2 at the same position: lure mass .004.
  **Reading: computing a variable's value erases its name's lexical prior from the state
  (bound number word → lure mass .004 at the answer); an unbound number word's value lingers
  as a weak signal (.118) and still never becomes the answer (behavioural lure 1.8%).**
  That is the "binding versus presence" contrast the level-4 control was designed for, with the
  sign the other way round from the plan's expectation.
- Level 4 bound target (incongruent@v2): value at its chain step .95, lure .013; at the end of
  its definition .46, lure .06; margin positive throughout.

## 2026-09-13 — storage

With the lead's approval, deleted the `hidden.npy` files of every `*_alt@*` group (72 files,
380 GB; never read by probes, patching recomputes activations) and the 0.9 GB smoke directory.
`probing/` on scratch went from 1.8 TB to 1.4 TB; the account's scratch total from 2.9 to 2.5 TB.
The per-user scratch limit behaves like ~3 TB. `20_run_model.py` no longer caches the control
groups' hidden states unless `--forced-all` is given. E33 resubmitted (393084 / 393085).

## 2026-09-13 — E33: the level-3 name effects were a demonstration artefact

Level 3, same 2,000 matched sets, demonstration seed 7 (used everywhere so far) vs seed 11:

| model | regime | demos | neutral | letters | queried: cong / inc / lure (base) | intermediate: cong / inc / lure (base) |
|---|---|---|---|---|---|---|
| 3B | chain | s7 | 96.6 | 98.7 | 95.1 / 94.6 / 0.4 (0.1) | 99.5 / 99.4 / 0.0 (0.1) |
| 3B | chain | s11 | **99.9** | 98.5 | 100 / 100 / 0.0 | 99.6 / 99.9 / 0.0 |
| 8B | chain | s7 | 85.7 | 96.6 | 82.5 / 83.6 / 0.7 (1.1) | 97.5 / 84.0 / 0.9 (0.9) |
| 8B | chain | s11 | **100** | 100 | 100 / 100 / 0.0 | 99.5 / 100 / 0.0 |
| 3B | direct | s7 | 13.4 | 25.8 | 18.9 / 13.9 / 8.6 (4.7) | 17.9 / 13.8 / 8.1 (4.7) |
| 3B | direct | s11 | 31.4 | 33.9 | 33.1 / 28.8 / 5.9 (5.5) | 31.9 / 29.3 / 7.0 (4.7) |
| 8B | direct | s7 | 41.3 | 37.9 | 48.8 / 44.4 / 3.5 (3.1) | 47.9 / 45.8 / 3.5 (3.5) |
| 8B | direct | s11 | 62.1 | 56.4 | 63.3 / 61.3 / 4.2 (3.0) | 63.7 / 61.8 / 3.6 (3.1) |

- With seed-11 demonstrations every level-3 chain condition is at 99.5–100% for both models.
  The 2-point interference, the +3 / +12 facilitation, the 8B operand-copy error (E26) and the
  "number word on the intermediate helps" effect (E26 question) all vanish. They were induced
  by the seed-7 demonstrations (one of which contains the operand coincidence).
- What is robust to the demonstration set: (i) under a chain the lure is never the answer;
  (ii) without a chain there is a small lure effect above baseline (3B: 5.9 / 7.0 vs 5.5 / 4.7;
  8B: 4.2 / 3.6 vs 3.0 / 3.1), 1–3 points; (iii) direct-answer accuracy itself swings by
  18–21 points with the demonstrations, dwarfing any name effect.
- Consequence for the paper: the demonstration set becomes a factor. Every behavioural number
  is reported as the mean (and range) over ≥ 3 demonstration sets, and no name effect is claimed
  unless it holds in every set (PREREGISTRATION amendment 2). The representational results
  (lure read only at name tokens; true value at the value step) were obtained under seed 7 and
  do not depend on error rates, but will be reproduced under one more seed.
- The level-4 and level-5 word-class effects (3B: 81.5 → 63 with a number-word queried name)
  must be re-tested under other demonstration sets before being believed.

## 2026-09-13 — E33 sweep: three demonstration sets (7, 11, 13), all levels, both models

Claim rule (amendment 2): sign holds in every seed AND pooled cluster-bootstrap CI excludes 0.
Full table: results/summary/seed_sweep_L{2,3,4,5}.json. What is claimable:

**Chain regime — the name's VALUE never matters.** Lure excess (lure rate minus pseudo-lure)
is within ±0.5 points in every cell of every level and model; congruent and incongruent move
together. What does move is the word CLASS by role: a number-word name on the queried
variable at level 4 costs the 3B model 10.5 points (interference +10.5 [9.4, 11.6]; facilitation
−10.9 [−12.0, −9.8]; both claimable, i.e. the same drop whether the value agrees or not); on the
intermediate at level 4, +3.5 / −2.7; on the deepest variable at level 5 a number word HELPS
(+7.3 facilitation, −3.5 interference, both claimable). The 8B model shows nothing under a chain
except seed-13 noise at level 4. Neutral accuracy itself varies 26–85 (3B, L4) and 43–100
(8B, L5) across demonstration sets.

**Direct regime — the name's value is used as a shortcut, in both directions.** Facilitation
on the queried variable is claimable almost everywhere: 3B +10.8 (L2), +5.4 (L3), +4.7 (L5);
8B +11.8 (L2), +4.8 (L3), +13.7 (L4), +20.2 (L5). Lure excess on the queried variable is
claimable at 3B L2/L3/L5 (+2.5 / +2.8 / +3.5) and 8B L3/L5 (+1.2 / **+20.5**); at 8B level 5
under seed-11 demonstrations the model answers the lure in 50.8% of incongruent cases above
baseline and gains 36 points from a congruent name: a full Stroop effect, induced by the
demonstrations, absent under a chain (8B L5 chain lure excess −0.3 / −0.1 / +0.1).

**Headline that survives:** without a chain, these models read a variable's value off its
name when they can (facilitation up to +20, lure answers up to +51 under some demonstrations);
with a chain, the name's value has no effect on the answer at any level, and the probes show
the lure is read only where the name sits and is erased at the value step. What a chain does
not remove is a word-class effect on equation selection (3B, level 4). The demonstrations
modulate everything and must be reported as a factor.

## 2026-09-13 — the direct-regime fallback, Llama-3.1-8B, level 5, seed-11 demonstrations

Neutral 12.8% (floor; answers "1" 34% of the time), congruent@v1 48.8%, incongruent@v1 5.7%
with the lure at **65%**: `eight=8`. When the model cannot compute the answer it echoes the
queried name's value. Nothing in the seed-11 demonstrations (answers 0, 4, 9) suggests this;
it is the fallback that a floor exposes. The same fallback is what the chain removes: 8B level
5 chain, seed 11, lure excess −1.1 / −0.4 / +0.4 (v1 / v2 / v3) at 43–100% accuracy.

## 2026-09-13 — representational replication under the second demonstration set (3B, level 3)

The probe map is unchanged by the demonstrations, unlike behaviour:

| regime | demos | end of definition (v2) | query (v2) | value step (v2) | answer (v2) |
|---|---|---|---|---|---|
| chain | s7 | acc .49 / ctl .17, lure .06 | acc .27 / **ctl .97**, lure .32 | acc .98, lure .00 | acc 1.00, lure .00 |
| chain | s11 | acc .36 / ctl .17, lure .06 | acc .23 / **ctl .97**, lure .40 | acc 1.00, lure .00 | acc 1.00, lure .00 |
| direct | s7 | acc .47, lure .06 | acc .32 / ctl .96, lure .43 | — | acc .80, lure .02 |
| direct | s11 | acc .35, lure .07 | acc .30 / ctl .96, lure .33 | — | acc .37, lure .04 |

Same for the queried variable (lure .11 at its definition, .51–.66 at the query with control
1.00, .00 at the value step and answer). **Conclusion: the three findings that carry the paper
— the lure is read only at name tokens, the query-position readout is name identity, and the
value step erases the lure — hold under both demonstration sets and in both regimes.** Only the
absolute accuracies move (3B direct answer-position probe .80 → .37 with the demos, tracking
task accuracy 13% → 31%).

## 2026-09-13 — E37: the chain null is an equivalence, not an absence of power

Two one-sided tests on the cluster bootstrap, smallest effect of interest δ = 2 points (the
size of the interference originally reported from a single demonstration set):

| | cells | equivalent to zero | claimable effect |
|---|---|---|---|
| chain of thought | 18 | **18** | 0 |
| direct answer | 18 | 10 | **6** |

Largest absolute lure excess in any chain cell, over four levels, two models and three
demonstration sets: **0.65 points**. The same design, on the same instances, detects effects up
to +20.5 points in the direct regime, so the null is not a power failure. This is the positive
control the null needed.

## 2026-09-14 — E34: the no-computation control refutes the pre-registered prediction

Level 1 puts a number-word name on a variable whose value is STATED (`six=3 + fish, fish=5`),
so resolving it needs no arithmetic. Amendment 3 predicted lure excess above δ there.

| model | regime | lure excess, stated variable | lure excess, queried variable | neutral acc (range over seeds) |
|---|---|---|---|---|
| 3B | chain | +0.3 | +0.1 | 97.0 [95, 100] |
| 8B | chain | +0.0 | +0.0 | 99.2 [98, 100] |
| 3B | direct | **+2.6*** | **+5.9*** | 46.3 [21, 93] |
| 8B | direct | +1.3 | **+11.2*** | 73.7 [38, 99] |

*claimable under amendment 2.

**The prediction failed, and the failure sharpens the claim.** Under a chain the lure is absent
even for a variable that requires no computation at all. So it is not computation that erases
the lexical prior: it is WRITING THE VALUE. The chain restates a stated variable
(`fish=5`) exactly as it writes a computed one, and both leave no lure. Without a chain,
where nothing is written, the lure appears for both roles and is roughly twice as strong on the
queried variable (the answer slot) as on the stated one.

Revised mechanism sentence for the paper: *the chain removes the lexical prior by writing the
value, not by computing it; the prior survives exactly where no value is written.*

## 2026-09-14 — E36 free-form chain: not answerable on base models

Given the instruction "Solve for the queried variable. Think step by step..." and no worked
examples, both base models degenerate into copying the problem line:

```
truck=2 + nine, nine=7 - 6; truck=?
truck=2 + nine, nine=7 - 6; truck=?   (x20)
```

Accuracy 16–22% in every condition at levels 3 and 5, with lure rates at zero because the
parsed "answer" is an echoed operand. Base models do not follow a zero-shot instruction; this
says nothing about the lure. **E36 is inconclusive for base models and is requeued for the
instruct models** (llama32-3b-it, llama31-8b-it, gemma3-4b-it, olmo2-7b-it), where the
instruction is in-distribution. If the instruct models also fail, the fallback is a
*prose-chain* regime: worked examples that reason in prose rather than in the equation format,
which tests format-dependence without requiring instruction-following.

## 2026-09-14 — E35 lure injection, and E36 on instruct models: both tier-A questions answered

**E35 (injection).** Transplanting the misleading name's activations at its own token positions
INTO a clean neutral run, 500 matched pairs, reading the final answer:

| model | regime | answer becomes the injected lure (all layers) | best single layer | accuracy before → after |
|---|---|---|---|---|
| 3B | chain | **0.000** | 0.002 (layer 1) | 1.00 → 1.00 |
| 8B | chain | **0.000** | 0.000 | 1.00 → 1.00 |
| 3B | direct | 0.026 | 0.026 (layer 0) | .156 → .148 |
| 8B | direct | 0.010 | 0.019 (layer 1) | .420 → .436 |

Prediction confirmed. Under a chain the name's representation can be transplanted wholesale and
the model still writes the right answer; the chain is causally insensitive to it. Without a
chain the same transplant shifts the answer to the injected number in 1–3% of cases, at the
earliest layers, the same locus the removal test found.

**E36 (free-form reasoning) now works, on instruct models.** No worked examples, only the
instruction; the models write prose reasoning ("## Step 1: First, we need to solve for the
variable \"ant\"...") and parse cleanly:

| model | neutral | name agrees | name lies | answers the name |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 71.0 | 80.1 | 69.6 | 0.4 |
| Llama-3.1-8B-Instruct | 67.0 | 71.4 | 67.2 | 0.0 |
| Gemma-3-4B-it | 84.9 | 80.4 | 82.4 | 0.8 |
| OLMo-2-7B-Instruct | 63.8 | 61.6 | 60.9 | 1.6 |

**The result does not depend on our prompt format.** With reasoning the models invent
themselves, at 61–85% accuracy (well off ceiling, unlike the forced format), the name's value
still never becomes the answer: at most 1.6%. Interference is within 1.5 points everywhere.
A congruent name still helps the 3B model (+9.1), which is the facilitation asymmetry again.

## 2026-09-14 — debug pass: two real bugs, and the one chain cell that is not equivalent

**Bug 1, the answer parser.** It read only the first line of a generation, so when a model
restated the problem before solving it the answer (on line 2) was scored as no answer. This hit
5–13% of instances in some groups and at *different rates per condition*, so it biased contrasts
rather than adding noise. Fixed to take the last `name=value` before the blank line that
separates few-shot problems. Generations are stored verbatim, so `scripts/61_reparse.py` re-scores
without a GPU: **36 of 1,068 groups changed, 485 of 2.1M rows.** Largest correction, Llama-3.2-3B
level 4 with a chain: incongruent@v1 62.8 → 65.8, congruent@v1 63.4 → 65.8, neutral 81.5 → 81.8.
The level-4 word-class effect therefore shrinks from ~19 to ~16 points but survives.

**Bug 2, colliding instance ids.** `data/L3_ident` reused `data/L3`'s instance ids, so any join
by id could cross datasets — which the re-parse tool itself did, briefly scoring every level-3
group at 0%. Ids now carry the naming scheme and the re-parse resolves names from the dataset the
run actually used.

**Bug 3, a silent default.** `56_seed_sweep.py` defaulted to two models, so regenerating after
the re-parse quietly shrank Table 1 and Figure 1 from eight models to two. The default is now the
whole panel, and `99_selfcheck.py` fails if a model has runs but is missing from the sweep.

**With the whole panel in the sweep, one chain cell is no longer equivalent to zero:**

| | cells | equivalent | with an effect |
|---|---|---|---|
| chain of thought | 30 | 29 | **1** |
| direct answer | 30 | 22 | 8 |

The exception is **OLMo-2-1B-Instruct**, level 3, number word on the intermediate variable:
lure excess +3.3 points [2.5, 4.1]. It is the weakest model in the panel and the only one that
cannot do the task with a chain: 47% neutral accuracy with word names against 77% with letters.
This is the prediction the "writing the value" account makes. The chain protects by writing each
value; a model that fails to compute the value never writes it, and the name survives. The paper
should say this rather than claim a universal null: **the protection is a consequence of solving
the problem, not of the format.**

## 2026-09-14 (evening) — paper, and E38 had never run

- **Paper.** The RQ1 paragraph now reports the equivalence test over the whole panel (29 of
  30 chain cells equivalent to zero at δ = 2; 8 of 30 direct cells with an effect) and names
  the exception (OLMo-2-1B-Instruct, level 3, intermediate variable, +3.3 [2.5, 4.1]) as the
  mechanism's prediction: its chain solves 47% with word names against 77% with letters, and a
  model that fails to compute a value never writes it. The conclusion says the protection
  comes from solving the problem, not from the format. Every number in that paragraph is a
  `\NUM` key from `51_tables.py` (the three that were hand-typed, 4.8 / 1.2 / 10.5, are gone;
  the level-4 word-class cost is now pooled over seeds: `wordclass-cost-L4`). Body back to
  4 pages after tightening related work and RQ2/RQ3.
- **E38 (identifier names) had never produced a result.** Every ident job failed at the
  layout guard: (a) the guard required single-token names, which the ident scheme violates by
  design (`q4` is two tokens); (b) demonstrations were drawn from the word pool, not from the
  dataset's `demo_words`. Fixed in `runner.py` (names must tokenize like the group's reference
  instance) and `20_run_model.py` (passes the manifest's demo pool). Verified with real
  tokenizers in a dev job (`scripts/12_ident_layout_check.py`) before the 11 held jobs go back
  into the queue.
- Worker queue: probe jobs ran before their cache jobs (no ordering) and failed; tasks can
  now declare `# needs: <path>` and are deferred until it exists.

## 2026-09-14 (night) — second family, level 3, Gemma-3-4B-it: the internals replicate

Cells picked by selectivity (rule 8), neutral-trained probes on the intermediate variable:

| position | Llama-3.2-3B: acc / sel / lure | Gemma-3-4B: acc / sel / lure |
|---|---|---|
| end of defining equation (P2) | .49 / .32 / .065 | .44 / .38 / .055 |
| value step (P4) | .98 / .66 / .000 | 1.00 / .90 / .000 |
| answer (P5) | 1.00 / .83 / .000 | 1.00 / .90 / .000 |

At the query the lure is readable in both models but the control task scores .94–.97 there:
name identity, not a value, exactly as in Llama. Patching (500 pairs): under a chain, injecting
the name's activations changes 0.0% of answers in Gemma (0.2% Llama) and there are no lure
errors to remove; without a chain the neutral patch removes 62% of Gemma's lure errors
(Llama 75%) with word-control damage 14% (Llama 22%, at floor accuracy), and injection lures
2.4% (Llama 2.6%). C3 and C4 therefore hold across families. OLMo-2-7B probes pending.

## 2026-09-14 (late) — accuracy pass on the paper: what was wrong

Every claim checked against its source. Corrected in the text (all numbers now `\NUM` keys):
1. "eight models on five levels" → eight at level 3, two at all five.
2. Fig. 1(b) caption: "every model inside the margin" was false on the left panel (OLMo-2-1B
   CoT loses 5.8 points to a congruent name on the queried variable); now says so.
3. Table 1 caption: "every starred effect is in the direct rows" was false (three starred CoT
   rows); now "starred CoT effects are within two points except OLMo-2-1B".
4. Limitations: "at most 8B" → 12B (Gemma-3-12B is in the panel).
5. Replication: the pre-CoT maxima quoted came from neutral-trained labelled probes; now from
   the letter-trained per-token cells (30.4 / 47.9 vs Kudo 17.8 / 33.2): positions match
   (equations 5 and 2), accuracies are higher than theirs, and the text says so.
6. Bound vs unbound lure mass compared different positions (L3 value step vs L4 answer); now
   both at the L4 answer position, best-accuracy layer: .004 vs .118, with the distractor's own
   accuracy (.04) stated.
7. `grid-layers` was a hard-coded default; now computed from the removal grid (in:v1 span,
   windows 0–7, up to 69% removed).
8. Llama-3.1-8B "lure rate at chance" contradicted its own starred +1.2; reworded.
9. "no lure errors / no answer changed" under CoT → at most 1 in 2,000 and 0.2% of 500.
10. **Irrelevant lure**: "does not produce lure answers" was false. Against the matched neutral
    baseline (`63_irrelevant.py`) a number word on the distractor is answered +2.1/+2.3 points
    above chance without CoT (both reliable), +1.0/+0.0 with CoT (neither reliable); accuracy
    moves ≤1.2 points. Presence alone is read as a value without a chain.
11. Equivalence test now over all five levels: 33 of 34 CoT cells equivalent (was 29/30 over
    levels 2–5); 11 of 34 direct cells with an effect.
12. Per-demonstration-set ranges added where a pooled number hid heterogeneity: level-5 lure
    excess 5–51 points; level-4 word-class cost 2–16 points.
13. Appendix promised content it does not contain (grids, lure-distance, teacher-forced rates,
    taxonomy) → "released with the code"; Appendix A now states that the Gemma/OLMo models ran
    on the Llama-verified word lists with the run-time guard confirming layout (0 exclusions in
    1,135 runs).
Verified unchanged: Table 1 (64 cells + stars vs sweep), Kudo's 17.8/33.2 (their Table 3),
figure instances (truck/four/3 in Fig. 1c; truck/nine/1 in Fig. 2), E26 wording, all refs,
labels and bib keys. Body: 4 pages. Self-check: 0 failures.

## 2026-09-15 — the exception gets a causal signature (E39)

OLMo-2-1B-Instruct internals at level 3. Patching under CoT **force-decodes the gold chain**, so
base accuracy is 97% even though the model's own chains are right only 47% of the time. Under
that correct chain, injecting the misleading name's activations into a neutral run moves the
answer to the lure in **2.7% of 500 pairs at layer 5** (of 16), with total damage 7.0% at that
layer. The neutral-word control at the same layer does 0.4%. Every other model is at 0.0–0.2%.
Of the damage the injection causes, 39% lands on the lure specifically, against 11% if the shift
were unspecific over ten digits.

Three independent measurements now agree for this model: behaviour (+3.3 lure excess under CoT),
the value-written control (+3.7 excess on the 1,207 instances whose chain wrote the correct
value), and patching (above). The exception is real, causal and localized; the paper says so and
still does not claim a mechanism for why the other seven models are immune.

Also corrected: the positional-copy control is now in the appendix with pooled numbers. Across
every model, level and demonstration set there are 4,073 CoT lure errors at the answer; the
number immediately before the answer is the lure in 13% and the true value in 11%. Neither
dominates, so positional copying does NOT account for the remaining CoT lure errors. (Per-cell
values at levels 1–2 reach 25–88% and are not representative.)

Operational: `# needs:` deferral in the worker queue only applies to workers started AFTER the
edit; bash has already parsed the loop in a running worker. That is why the OLMo probe job ran
before its cache job and failed. Documented at the top of `scripts/slurm/worker.sh`.

## 2026-09-15 — what actually predicts the protection: representation, not the written token

OLMo-2-1B CoT probes finished. At the step where the chain writes the INTERMEDIATE variable's
value, that model's probe reaches accuracy 0.22 (selectivity +0.07, lure at chance .095) — the
value is not linearly represented there. For the QUERIED variable the same model reaches 1.00.
Across the 8 (model, target) cells with probes:

| value decodable at the writing step | cells | behavioural lure excess |
|---|---|---|
| yes (accuracy ≥ .98) | 7 | all within ±0.5 pts |
| no (accuracy .22) | 1 | +3.3 pts, the exception |

The alignment is role-specific: OLMo-2-1B's behavioural exception is on v2 (+3.3, reliable) and
not on v1 (+0.5, not reliable), exactly matching where its probe fails. Together with the
injection result (answer moves to the lure at layer 5 under a forced correct chain), this gives
a coherent account that survives the control that killed the earlier one: **writing the value as
a token is not sufficient; the model must also hold the value at that step.** The paper states
this as an alignment over 8 cells, not as a causal claim.

Also: `92_page_budget.py` was too lenient (a heading within the first 4% of a page counted as a
clean break, so 3 lines of conclusion on page 5 still reported "body ends on page 4"). Now any
body text above the heading counts. Body is genuinely 4 pages.

## 2026-09-15 — E38 (identifier names): the cue is specific to number words

`q4` in place of `four`, level 3, four models, three demonstration sets. Compared with the word
scheme on the same four models:

| scheme | direct-regime lure excess (4 cells) | reliable |
|---|---|---|
| number words (`four`) | -0.2 to +2.8 pts | 3 of 4 |
| identifiers (`q4`) | -0.2 to +0.2 pts | 0 of 4 |

So a digit inside an identifier is **not** read as the variable's value, even without a chain,
while a number word is. The pre-registration expected the identifier might be a *stronger* cue
(an explicit digit rather than a lexical association); it is weaker, and in fact absent.
Accuracy still moves under the identifier scheme (facilitation +8.2 for Gemma-3-4B, -2.3 for
Llama-3.2-3B, both reliable, no consistent sign), so the name is not ignored; its digit simply
never becomes the answer. This bounds the paper's claim to lexical priors and is now a Results
paragraph.

**All GPU experiments for the paper are complete as of this entry.**
