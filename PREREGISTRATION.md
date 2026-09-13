# Pre-registration

*Committed 2026-09-12, before any model was run on this dataset.* Anything below may be changed
**only** by adding a dated amendment at the bottom saying what changed and why.

## 1. Primary level, regimes, models

- Main level: **3** (`v1=d±v2, v2=d±d; v1=?`, two steps, forward reference), Kudo et al.'s
  main level for cross-model comparison. Levels 2 and 5 are robustness; level 4 carries the
  irrelevant-lure control only.
- Regimes: chain of thought (CoT) and direct answer, three fixed same-level demonstrations each.
- Core models: Llama-3.2-3B and Llama-3.1-8B (base), the two Kudo et al. models available
  under the scope rule. The go/no-go on **Oct 1** is decided on these two.

## 2. Primary contrasts (RQ1, behaviour)

Within matched sets, cluster-bootstrap 95% CI over sets, per model × regime × target position
(intermediate `v2`, queried `v1`):

| contrast | name | answers |
|---|---|---|
| neutral − incongruent accuracy | interference | do misleading names hurt? |
| congruent − neutral accuracy | facilitation | do matching names help? |
| lure rate in incongruent | lure rate | are the errors the lure specifically? |
| interference(direct) − interference(CoT) | CoT protection | does writing the steps reduce it? |
| lure rate(irrelevant) vs (incongruent) at level 4 | binding vs presence | is a number word enough, or must it be bound to a chain variable? |

Model-level test: logistic regression `correct ~ condition × regime` with cluster-robust SEs
on set id (mixed random intercept per set in the appendix).

## 3. Representation (RQ2)

- Probes: linear, trained on `train_neutral` only, Kudo recipe (SGD 1e-3, 10k epochs, full
  batch), seeds {0,1,2}, all layers, positions `end@r`, `query`, `cotpre@r`, `anspre`
  (`def@r` reference only).
- Metrics per (position, layer): true-value accuracy; lure rate; lure mass; margin
  log p(true) − log p(lure). **Crossover step** = first position in the order
  `end@r → query → cotpre@r → anspre` at which the layer-max mean margin is positive and stays
  positive.
- Headline claim is about the crossover step and the margin at `anspre`, CoT vs direct.
- Selectivity threshold: a probe cell is reported only if accuracy − control accuracy ≥ 0.3.
- FDR (Benjamini–Hochberg, α = 0.05) across all layer × position tests of margin > 0.

## 4. Instance-level link

Logistic regression: `lure_error ~ margin(end@r) + margin(query) + margin(cotpre@r)` at the
best neutral-accuracy layer per position, cluster-robust on set id, with model and level as
covariates when pooled.

## 5. Causation (RQ3)

- `main`: patch residual stream at every token carrying the target name, source = matched
  neutral twin, destination = incongruent; sweep single layers, 4-layer windows, all layers.
  Reads: the next token at `cotpre@r` and at `anspre` under teacher forcing.
- **Recovery** = P(correct after patch | lure error before), reported at the best layer set
  and at all layers.
- Controls: `ctl_word` (neutral_alt → neutral; damage rate must stay < 5% at the reported
  layer set, else the recovery is not interpretable) and `ctl_lure` (incongruent_alt →
  incongruent; the answer following the *new* lure is the positive control for the name's
  causal role).

## 6. Exclusions, fixed in advance

- Instances that break the token layout (multi-token name or shifted positions) are excluded
  and counted; a group with > 1% exclusions fails the job.
- A model with < 90% neutral accuracy at level 3 under CoT does not enter Table 1 for that
  level (it cannot show interference it does not have the accuracy to lose); it is reported in
  the appendix.
- Free-generation answers are parsed as the last `name=digit` on the first line; unparseable
  outputs count as wrong and not as lure.

## 7. What each outcome means (PLAN.md §5)

Filled in advance so the introduction does not depend on the result: lure fades + correct →
computation overrides the name; lure fades + lure answer → late readout error; lure persists +
correct → hidden interference; lure persists + lure answer → faithful computation from a
corrupted value.

---
*Amendments below this line, dated.*

**Amendment 1 (2026-09-12, after the literature read, before any model run).** (a) Add E22, a
positional-copy control after Liu (2026): the "late readout" cell is claimed only if lure errors
at P5 occur when the trailing chain number is the true value. (b) Add the Geirhos-style lure
index (lure / (lure + true), congruent trials excluded) as a reported column beside accuracy.
(c) Patching is additionally scored with Zhang and Nanda's normalised logit difference between
true value and lure, beside the recovery rate; both must agree for a layer set to be reported.
None of these change a primary contrast.

**Amendment 2 (2026-09-13, after E33).** The fixed three-shot demonstration set proved to be a
large source of variance: at level 3 the 8B model's chain accuracy moved from 85.7% to 100%
and every name effect vanished when the demonstrations changed. From now on: (a) every
behavioural quantity is computed for at least three demonstration seeds (7, 11, 13) and
reported as the mean with the across-seed range; (b) a name effect (interference, facilitation,
lure rate above baseline) is claimed only if its sign holds in every seed and the pooled
cluster-bootstrap CI (clustering on matched set, seeds as replicates) excludes zero; (c) the
seed-7 results obtained before this amendment are kept and reported as one of the three seeds.
The representational and causal analyses stay on seed 7 plus one replication seed.
