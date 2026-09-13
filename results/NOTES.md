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
