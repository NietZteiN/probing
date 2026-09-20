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

**Amendment 3 (2026-09-13, before the tier-A runs).** The headline claim is now a null under
chain of thought, so it is tested as an equivalence rather than asserted from a non-significant
difference. The smallest effect of interest is δ = 2 percentage points of lure excess, fixed
here before the level-1 and free-form runs. A chain cell counts as equivalent when its 90%
cluster-bootstrap interval lies entirely inside [−δ, +δ]. The direct regime serves as the
positive control: the design must detect an effect there on the same instances. Additional
pre-registered experiments: **E34** no-computation control (level 1: the intermediate's value is
stated, not computed; if the lure intrudes only there, computation is what erases it), **E35**
lure injection (patch the incongruent name's activations into a neutral run and read the chain's
value step; a positive causal test in place of the removal test, which has no errors to remove),
**E36** free-form chain (no worked examples, an instruction only, free-text answer parsing).
Predictions recorded before seeing the results: E34 lure excess above δ at the stated variable
and within δ at the computed one; E35 injection rate near zero at the value step; E36 lure
excess within δ, i.e. the format is not what removes the lure.

**Amendment 4 (2026-09-18, before any run of the regime).** E31, the value-only chain
(`simple`: demonstrations and model write `cup=5, pen=6`, the value steps of the chain without
the equations; Kudo et al.'s "Simple CoT", Table 6). This is the arithmetic counterpart of the
code study's terse trace (`v = 3`), whose contamination the code study located at the readout.
The full chain used everywhere else in this paper writes the expression before every value, so
it corresponds to the code study's `trace_expr`. Predictions, fixed here: (i) at level 3 the
written-lure excess at the target's value step (the model writes `name=<lure>` for the
number-word variable, against the matched neutral twin's rate of writing that digit for the
same role) exceeds δ = 2 points for at least Llama-3.2-3B and Llama-3.1-8B, on the intermediate
variable at least; (ii) for the same models, a neutral-trained probe at `cotpre@r` in the simple
regime reads the true value at ≥ 0.85, so the failure, if present, is one of readout and not of
representation, as in code. Refutation: (i) false for both Llama models means the value-only
format is dangerous in code and not in arithmetic, and the merged paper must say why rather than
claim symmetry. Analysis script fixed in advance: `65_simple_chain.py` (written-lure excess at
the value step per role, three demonstration seeds, cluster bootstrap, the paper's claim rule).

**Amendment 5 (2026-09-20, before any of these runs).** Model-kind axis. The panel so far varies
family and size; it does not vary *how* a model was tuned. We add four models on one spine,
Llama-3.1-8B-Instruct, so that tokenizer, word pool and token positions are identical and every
row is comparable instance by instance: a single-task LoRA finetune, a joint multi-task LoRA
finetune, a TIES merge of six single-task adapters (all three obtune artefacts, folded into the
weights at load time), and a reasoning-tuned checkpoint on the same spine
(`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`). With the base and instruct models already in the panel
this gives the ladder base → instruct → finetune → multi-task → merge → reasoning.

Prediction, fixed here: the direct-regime lure excess does NOT differ beyond the equivalence
margin δ = 2 between instruct and any of the three tuned variants, because LoRA finetuning on a
code/task mixture changes what the model knows, not whether it reads a variable's value off its
name. The reasoning model is a separate case and no prediction is fixed for it, since its output
format differs and its regime has to be chosen (below). Refutation of the tuned prediction: any
tuned variant differing from `llama31-8b-it` by more than δ under the claim rule, which would
say that the effect is a property of the tuning mixture rather than of the architecture or the
pretraining.

Analysis fixed in advance: `68_model_kind.py`, reporting lure excess over the matched neutral
twin with the paper's claim rule (same sign in all three demonstration seeds, pooled cluster
bootstrap excluding zero). Reasoning models emit a thinking prefix before the answer, so the
few-shot `direct` and `cot` regimes may not apply to them; if they do not, the reasoning row is
reported in the `free` regime only and is labelled as not directly comparable to the rest of the
ladder rather than being forced into it.
