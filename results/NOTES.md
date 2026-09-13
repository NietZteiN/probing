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
