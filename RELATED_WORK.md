# Related work, and what is left for this paper to claim

*Rewritten 2026-09-12 after reading every paper in [`papers/`](papers/) (39 PDFs, indexed in
`papers/SOURCES.tsv`). Page numbers refer to the PDF version on disk; "(p. N)" claims were read
from the text, not from abstracts. The four sets of raw reading notes this was written from are
kept in the session scratchpad; the claims below are the ones that survived cross-checking.*

**Short answer to "has this been done?": no, but two papers from mid-2026 are close enough that
they must be cited in the first paragraph of related work, not the last.** Wang (2026) runs a
matched-neutral-twin Stroop paradigm with neutral-to-conflict residual patching; Hu et al. (2026)
run a congruent/incongruent verbal conflict task and interpret it as in-weight versus in-context
competition. Neither measures errors (Hu et al. report 100% accuracy in both conditions), neither
has a chain of thought, and neither probes hidden states across reasoning steps. Liu (2026) shows
that small models' arithmetic readout is a positional copy of the last chain number, which is both
a risk to our "late readout" outcome cell and a control we now have to run (E22).

---

## 1. The paper this one builds on

**Kudo, Aoki, Kuribayashi, Sone, Taniguchi, Brassard, Sakaguchi, Inui (2026). LLMs Faithfully
and Iteratively Compute Answers During CoT: A Systematic Analysis With Multi-step Arithmetics.
Findings of EACL 2026, pp. 1114–1153.** `papers/kudo2026faithful.pdf`; code
github.com/keitokudo/faithful-cot-multistep-arithmetic (fn. 1, p. 2).

Read in full. Synthetic `A=1+B, B=2+3; A=?` problems at five levels (Table 1, p. 3), digits 0–9,
values single-digit, three same-level demonstrations, 10,000 probe-train / 2,000 test with
equation-level disjointness (fn. 3, p. 2). One linear probe per (position, layer, variable),
SGD lr 1e-3, full batch, 10,000 epochs (Table 8). Every sub-answer that needs a computation step
becomes decodable only after the chain begins: t*_eq > 0 at τ = 0.9 (Table 2, p. 4; Table 3
across nine models, p. 6); pre-CoT accuracy peaks around 60% in the second half of the defining
equation (p. 4). Activation patching in an (equation × 4-layer) grid shows the final answer
depends causally on the chain, barely on the input (Fig. 5, p. 7), with strong recency bias
(Fig. 6, p. 8). Llama-3.2-3B is the weakest model: 93.2 / 90.9 / 38.5% at levels 3–5 (Table 7,
p. 13). The implicit (no-chain) format reaches 77.8% with no position above τ (Table 6, p. 13).

**What they cannot say.** Every variable name is a letter, so a decoded value has exactly one
possible source. Nothing in the design distinguishes "the model computed 5" from "the model read
5 off a cue", because no cue ever disagrees. We keep their levels, prompt structure and probe
recipe, and add the disagreement. Their conclusion is extended, not challenged: we ask what the
on-the-fly computation is sensitive to. Note for the paper: our operators are spaced
(`1 + two`) because the Llama-3 tokenizer merges `-` with a following word; absolute token
positions are therefore not comparable to their figures, only equation-level ones.

## 2. Cue conflict in language models (the closest work)

**Wang, Han-yu (2026). Persistent Priors, Preserved Targets: A Stroop-Style Paradigm for
Lexical Override. arXiv:2606.07555v5 (8 Aug 2026).** `papers/stroop2026priors.pdf`.
A conflict prompt defines *doctor* as *forest* and asks for a related word; the distractor is
the familiar associate *hospital*; a neutral control swaps *doctor* for a semantically weak
word while keeping target and distractor fixed (pp. 2–3). Metric: teacher-forced log-prob
margin, interference Δ = neutral − conflict. Eleven settings (Qwen2.5, Gemma-2, OLMo-1B,
Mistral; 1–9B), four conflict families. Every model shows positive interference, Δ from 1.31 to
2.71 nats (p. 4; Table 8, p. 12); prior strength predicts interference with slope +0.114
(p < 1e-9) except for antonyms (Table 1, p. 5); instruction tuning and size reduce it (p. 5).
Patching neutral activations into the conflict prompt at three positions restores the margin
almost fully, R = 0.92–1.06 (Table 2, p. 6); donor-item patches reduce recovery in all 142
cases (Table 14, p. 14). The author concedes that patching the defined word and the query word
"approaches replacement of the input difference" (pp. 6–7; App. F).

*Difference.* Same matched-twin logic and same patching idea, but on lexical semantics scored as
a single next-token margin: no generation, no accuracy, no chain of thought, no probes over
reasoning steps, mechanistic results only for antonyms in 1–2B models (pp. 8–9). Our conflict is
a numeric binding that must be *used* in multi-step arithmetic; we measure errors and lure rate
with and without a chain, read true-value versus lure from hidden states at P2–P5, and patch the
name occurrences with two controls. Cite as establishing that a prior persists after an explicit
local definition; position ours as asking whether the persistence survives a reasoning chain and
whether it produces errors rather than margin loss.

**Hu, Angstadt, Storks, Huang, Taxali, Weigard, Lewis, Sripada (2026). Conflict and Congruency
Effects in Large Language Models: In-Weight and In-Context Competition in a Verbal Conflict
Task. arXiv:2608.11510 (11 Aug 2026; the PDF on disk carries no arXiv stamp, verify version).**
`papers/stroop2026conflict.pdf`. Two if–then colour rules then "The crayon is blue, so you say
the crayon is"; congruent rules map each colour to itself. Gemma-2-2B and six Pythia models.
Congruency effect 0.397 in Δ-probability, but "Accuracy was 100% in both conditions" (p. 11).
Attribution graphs and attention ablation locate a short-range pathway to the colour cue in
congruent prompts and a long-range pathway to the rule in incongruent ones (pp. 12–13);
fine-tuning that strengthens the default hurts incongruent trials only (p. 14).

*Difference.* Their prior is an identity mapping and the override is a rule table read off a
single next token in a task nobody gets wrong. Borrow the in-weight versus in-context vocabulary
and the congruent/incongruent design; state that neither error behaviour nor hidden-state
content across reasoning steps was examined.

**Geirhos, Rubisch, Michaelis, Bethge, Wichmann, Brendel (2019). ImageNet-trained CNNs are
biased towards texture. ICLR 2019.** The design we transfer. Cue-conflict images fuse a shape
and a texture by style transfer, both pre-selected to be classified correctly alone (p. 4;
App. A.6, p. 15); congruent pairs are excluded from the bias estimate (p. 15); shape bias =
shape decisions / (shape + texture decisions), computed only over trials whose answer matches one
of the two cues, with overall accuracy reported separately (Fig. 4, p. 6). Humans 95.9% shape,
ResNet-50 22.1% (p. 5). *For us:* the neutral and congruent twins play the pre-selection role;
report a lure index = lure answers / (lure + true answers) beside accuracy (added to E3).
**Stroop (1935)** is the origin of the vocabulary (interference, facilitation).

## 3. Misleading names in code (the motivation)

**Le, Nguyen, Nguyen (2026). Do Machines Struggle Where Humans Do? LLM and Human Comprehension
of Obfuscated Code. arXiv:2606.31725v1 (30 Jun 2026).** `papers/obf2026humans.pdf`. Cite in the
third person under review. Output prediction across obfuscation tiers L0–L3 on 0.5–8B models;
L1b is misleading renaming ("adversarial renaming" is their label; we say misleading). For the
strongest models misleading names hurt more than uninformative ones: DS-R1-Qwen-7B 63.8 → 64.2
→ 51.8 (L0 → L1 → L1b), SmolLM3-3B 45.0 → 44.0 → 34.2 (Table II, p. 4); human experts *improve*
at L1b, which no model reproduces (pp. 4–5, 7). **The "21% / 10%" figures in PLAN.md are
correct but scoped:** they describe the n = 80 tail subset where top-20% semantic displacement
and top-20% identifier surprisal co-occur (accuracy 21.25%, high-confidence-incorrect 10.00%,
against a 31.68% baseline subset; Table X, p. 9), not L1b overall. The draft now says
"impairs comprehension through semantic displacement and interference" and cites the scoped
numbers only in prose that names the subset. Purely behavioural; no congruent condition; many
identifiers change at once.

**Le, Pham, Van, Phan, Phan, Nguyen (2025). When Names Disappear. arXiv:2510.03178.** Frontier
models on ClassEval and LiveCodeBench under four name-only obfuscations. Summarisation collapses
(87.3 → 58.7 for GPT-4o, p. 5); execution prediction also drops, but *misleading* names are
often the mildest perturbation: GPT-4o LiveCodeBench 82.9 → 82.3 (misleading) versus 68.7
(ambiguous) (Table 3, p. 7). Names act as retrieval keys for memorised outputs (Table 4, p. 9).
*Caution for us:* "misleading" must be operationalised with a concrete competing value, as we
do; a name that merely suggests a wrong behaviour need not hurt a frontier model.

**Lam, Wang, Huang, Lyu (2025). CodeCrash. NeurIPS 2025 (arXiv:2504.14119v3).** Misleading
comments, prints and hints in code; 17 models. Direct inference −23.2% average, CoT −13.8%, with
the largest relief for hints (−20.1 → −6.2) but no elimination "particularly for small
open-source models": Llama-3.1-8B −19.8% and Qwen2.5-7B −19.7% under CoT (Tables 1–2, pp. 4–5).
Renaming alone is mild, −4.0% on CRUXEval (Table 10, p. 17). Failure traces show the model
"absorbs the misleading messages directly into its reasoning" (p. 5). *For us:* their cue is a
wrong answer written in the context; ours is never written anywhere. Their "CoT reduces but does
not remove" pattern at 7–8B is the behavioural shape our regime contrast tests representationally.

**Orvalho, Kwiatkowska (2025). Are LLMs Robust in Understanding Code Against
Semantics-Preserving Mutations? arXiv:2505.10443v3.** Renaming to random identifiers *raises*
accuracy for most open models (Qwen2.5-Coder 62.6 → 76.6, Table 2, p. 8), and 10–55% of
correct answers rest on flawed reasoning (Table 1, p. 7). Closest to our neutral condition;
cannot separate losing a helpful cue from acquiring a competing one.

**Gao, Gao, Wang, Sun, Lo, Yu (2023). Two Sides of the Same Coin. ICSE 2023.** Splits an
identifier's influence into a direct (spurious) and an indirect (through code semantics) effect
and removes the direct effect by logit subtraction (CREAM, Eq. 18, p. 6). One rename flips a
function-naming model from "sort" to "open" (Fig. 1, p. 1). The conceptual split is ours: the
direct effect is the lure. **Yang, Shi, He, Lo (2022). Natural Attack. ICSE 2022.** Searched
renamings that flip CodeBERT (53.6% success on vulnerability prediction, Table 2, p. 8); names
are chosen to be semantically *close*, nearer our congruent than our incongruent condition. Cite
to mark the difference between a searched perturbation and a designed cue conflict.

## 4. Chain-of-thought faithfulness measured from the outside

**Turpin, Michael, Perez, Bowman (2023). NeurIPS 2023.** Unverbalised biasing features move
answers: of 426 explanations supporting biased predictions, one mentions the bias (p. 3);
zero-shot CoT accuracy drops up to 36.3 points (p. 6); the accuracy drop is a valid metric
because almost all of it is bias-consistent (App. F.5). They point to interpretability tools to
check whether the model registers the cue (p. 9). **Lanham et al. (2023). arXiv:2307.13702.**
Truncation and corruption of the model's own chain; faithfulness worsens with size from 13B
to 175B (p. 7); on synthetic addition, same-answer-without-CoT rises with size and ease (p. 8);
they lack "a separate way by which to understand the model's real internal reasoning process"
(p. 9). **Chen et al. (2025). arXiv:2505.05410.** Reasoning models verbalise a hint they used
25–39% of the time (p. 6); their conclusion asks for "probing the model's internal activations"
(p. 12). **Tutek, Hashemi Chaleshtori, Marasović, Belinkov (2025). EMNLP 2025.** Unlearn a
reasoning step from the parameters and re-ask without CoT; context perturbation "does not remove
knowledge from parameters" (p. 1); flips 39.6% vs 16.2% for add-mistake on ARC (Table 2,
p. 7); they set aside "procedural information driving arithmetic reasoning" (p. 10).
**Lewis-Lim, Tan, Zhao, Aletras (2025). arXiv:2508.19827.** Instruction-tuned 7–8B models fix
the answer before the chain (change it 25% of the time) with flat confidence trajectories,
faithfulness ≈ 0 under Turpin-style cues (Table 5, p. 13). **Wang et al. (2025). Chain-of-Probe.
Findings of NAACL 2025.** Early-answer rate 0.43–0.52 on MMLU (p. 2589); 17–27% right answers on
wrong chains (p. 2592).

*Difference, stated once.* All of these infer reliance from answer changes on closed or
black-box evaluations. We measure it inside the model under a controlled conflicting cue whose
correct and lure values are both known, in the arithmetic regime Lanham et al. identify as the
one of highest CoT reliance, with a matched twin for causation.

## 5. Chain-of-thought internals on arithmetic

**Ye, Xu, Li, Allen-Zhu (2025). Physics of LMs 2.1. ICLR 2025.** From-scratch models on iGSM;
V-probes read necessity, dependency and value at the end of every solution sentence (p. 10),
with a random-init control; values ≈100% decodable at sentence ends; mistakes are visible in
the state before generation, e.g. wrongly written unnecessary parameters are mis-probed as
necessary (Fig. 8, p. 12). Our closest methodological precedent; our late-readout cell (state
right, output wrong) is the complement of their "mistakes are already in the state".

**Liu, Ming (2026). The Readout Shortcut. arXiv:2605.22870v1 (20 May 2026).**
`papers/readout2026shortcut.pdf`. In 1–3B instruction-tuned models on GSM8K the final answer is
a copy of whichever number sits in the trailing answer-context slot: with gold kept and
everything else corrupted accuracy is 1.00/.991/.580, with gold corrupted .079/.117/.037
(Table 1, p. 3); in free generation the answer equals the last chain number 96–97% of the time
and accuracy given gold-not-last is 0.4–3% (Table 2, p. 4); a wrong trailing number is worse
than none (Table 3, p. 4). At 7–8B the copy remains (Δcopy ≥ .80) but distractor following drops
to 5–32% (Table 7, p. 7). Replacing numbers with English words costs −2.4% (App. R, p. 16).
Scope statement: "We make no claim about generation-time computation—only that the readout does
not faithfully use it" (p. 2).

*Consequence for us.* If our chain writes `two=5 … pen=1 + 5, pen=6` and the answer is still the
lure, that is a violation of Liu's copy gate induced by a lexical cue, and it is the interesting
case. But to claim a "late readout" cell we must show the lure at P5 is not positional copying:
**E22** reports, for lure errors at P5, how often the trailing chain number was the true value
(then the copy account cannot explain the error) versus the lure (then it can). His 7–8B results
are the regime where content-selective readout emerges, which is our range; his App. R
motivates a number-*word* lure.

**Yu (2025). arXiv:2411.15862.** Value probes at the last prompt token without CoT in Qwen2.5-72B:
first and last step decodable, intermediates "hardly detected" (p. 3); no-CoT accuracy collapses
under format changes (Table 1, p. 4) while explicit CoT stays near 100%. Precedent for our P3
read in the direct regime. **Lin, Xie, Yuan, Yang (2025). Implicit Reasoning in Transformers is
Reasoning through Shortcuts. arXiv:2503.07604v3.** A from-scratch model and GPT-4o both chain
numerals off the surface without binding them to variables, failing when a variable is a
subtrahend (GPT-4o ≈100% → 30%, Fig. 7, p. 8); patching hits the variable token only for
`number − variable` (App. G, p. 16). Our lure exploits the same surface-reading from the other
side: a name that *is* a numeral. **Mehrafarin, Parekh, Konstas (2026). arXiv:2604.23351v3.**
Patching a single CoT-run state into a direct-answer run of Llama-3.1-8B / Qwen2.5-7B recovers
the answer for 99.5% of GSM8K items, including chains that ended wrong (Table 2, p. 4); only
20.7% of successes survive a random-vector control (p. 5). Latent correct signal in wrong chains,
same models as ours. **Somov et al. (2026). Breaking the Chain. arXiv:2603.16475v2.** Structured
intermediates are "influential context rather than stable causal mediators" (p. 1); the gap
between identity and counterfactual edits is 18–23 points (p. 4) and larger models bypass the
trace when the input allows a direct route (p. 7). **Prabhakar, Griffiths, McCoy (2024).
arXiv:2407.01687v2.** GPT-4 overrides a correct shift-cipher chain with a higher-probability
word: correct chain → wrong answer 7–14%, wrong chain → right answer up to 55% for
high-probability words (Table 1, p. 7). The clearest behavioural precedent for a prior
overriding a correct chain at the answer; ours is a controlled token-level prior with the
override localised. **Kudo et al.** also report that the correct value often appeared earlier in
the chain of instances Llama-3.2-3B got wrong (Fig. 3, p. 6).

*Difference, stated once.* None of these has matched triplets that differ only in whether a
variable *name* lexically suggests a wrong value, probes that separate "the state holds the
lure" from "the state holds the truth but the output uses the lure", or a twin patch showing the
name is the cause. Closest on outcome: Liu; closest on method: Ye et al.

## 6. Binding and number representations

**Feng, Steinhardt (2024). ICLR 2024.** Binding IDs are additive vectors on entity and
attribute tokens; mean interventions flip retrieval with accuracy 0.99 → 0.00 (Table 1, p. 7);
fidelity increases with scale and is weakest in small models (Fig. 6, p. 8); the token after the
entity also carries binding (p. 5). No parametric prior competes with the in-context binding.
**Wu, Geiger, Millière (2025). ICML 2025.** A 38M model trained from scratch on 17-line
`a = 1 / b = a / … / #c:` programs (App. B, p. 13): no arithmetic, single-letter names, no
number words, no CoT. Three learning phases; the final mechanism is built *on top of*
line-position heuristics that never disappear (Fig. 4e, pp. 7–8), and full-state linear probes
fail (30.9% per variable, Table 1, p. 15) even where causal tracing succeeds. Surface similarity
of the task only; cite for shortcut-plus-mechanism coexistence and for the probe caution.
**Gur-Arieh, Geva, Geiger (2026). Mixing Mechanisms. ICLR 2026.** Retrieval of a bound entity in
nine 2–72B models is a mixture of positional, lexical and reflexive signals; positional retrieval
explains "only 20%" of behaviour in middle groups while the other two "produce sharp, one-hot
peaks" (p. 7). **Terminology trap:** their "lexical mechanism" is retrieval keyed on the identity
of the query token (p. 5), *not* a token's pretrained meaning bypassing context. Our text must
say "lexical lure" or "lexical prior" and never imply their mechanism is ours. Their 1-hop task
has no arithmetic, no CoT and no conflict; our level-4 control separates presence of a number
word from a conflict at the queried binding, which their design lacks.

**Stolfo, Belinkov, Sachan (2023). EMNLP 2023.** Number *words* share the late-MLP arithmetic
circuitry with digits (GPT-J relative indirect effect 27.8% for words vs 40.2% for digits, Table
1, p. 7; 50% neuron overlap at layer 19, Fig. 7, p. 8), which makes a number-word name a
plausible competitor for the arithmetic pathway. **Zhu, Dai, Sui (2025). COLING 2025.** Number
values are linearly decodable from the first layer (ρ ≈ 1), best in intermediate layers (p. 3,
5), but decodable at most positions without being *used*: patching non-number tokens has
"almost zero effect" (p. 7). Justifies linear probes and is why decodability alone is never our
evidence. **Heinzerling, Inui (2024). ACL 2024.** A token's parametric numeric attribute is a
2–6-dimensional, causally effective direction at the entity token in early-middle layers
(pp. 4–8), and in 7B models the numeric subspace is generic across properties (pp. 8–9): a
mechanistic reason a number word could inject its stored value. **Nikankin, Reusch, Mueller,
Belinkov (2025). ICLR 2025** (bag of heuristics) and **Rai, Yao (2024). ACL 2024** (neurons
holding partial results during CoT arithmetic) are cited for completeness in the appendix.

## 7. Distraction and surface cues in arithmetic

**Shi et al. (2023). ICML 2023.** GSM-IC: one irrelevant sentence drops macro accuracy to 6%
(Table 3, p. 6); lexical overlap with the problem matters more than the distractor's number
(p. 2; Table 4, p. 8). Supports manipulating the name rather than the value. **Mirzadeh et al.
(2025). GSM-Symbolic. ICLR 2025.** Proper-name changes are nearly harmless (Gemma2-9b 87.0 →
88.6) while number changes hurt (→ 83.1) (Fig. 4, p. 7) and an irrelevant numeric clause costs
up to 65 points (Fig. 8, p. 10). Their names are never number words, so their taxonomy cannot
produce a lexical–numeric conflict; ours is exactly the missing cell. **Yang, Kassner,
Gribovskaya, Riedel, Geva (2025). Findings of ACL 2025** and **Arcuschin et al. (2025).
arXiv:2503.08679** are shortcut and in-the-wild-unfaithfulness references for the appendix.

## 8. Methods we follow

**Zhang, Nanda (2024). ICLR 2024.** Recommendations we adopt and cite: corrupt with an
in-distribution, equal-length token swap rather than noise (p. 3, 8), which our neutral twin and
both control patches are by construction; report a normalised logit difference between the
correct and the competing answer rather than probability (p. 8; Eq. p. 3), so our patching
metric is (LD_patched − LD_incongruent)/(LD_neutral − LD_incongruent) with LD = logit(true) −
logit(lure), beside the recovery rate; single layers before windows, windows read "as the joint
effects of the full window" (p. 9); try alternative corruptions (p. 9), which our two controls
are. **Hewitt, Liang (2019). EMNLP 2019.** Control task: output is a deterministic function of the
word type, sampled independently at random per type with matched marginals (§2, p. 3);
selectivity = task accuracy − control accuracy (p. 2); linear probes are the most selective but
still memorise (71.2% control accuracy on PoS, Table 1, p. 5). Ours: type = the variable name,
label = a random digit per name; report the ceiling of p. 4. **Pan et al. (2026). Survey,
arXiv:2601.14270.** Frames the field's demand as moving "beyond correlational analysis, which
only proves information presence, to rigorous causal verification" (p. 9), which is the
probe-then-patch structure of this study.

## 9. Positioning table (for the paper's half page)

| area | representative | this paper differs by |
|---|---|---|
| CoT internals on arithmetic | Kudo et al. 2026; Ye et al. 2025 | adding cue conflict, so the decoded value has two possible sources |
| Cue conflict in LMs | Wang 2026; Hu et al. 2026 | a numeric binding used in a reasoning chain; errors, not margins; probes over steps |
| Readout shortcuts | Liu 2026; Prabhakar et al. 2024 | a controlled lexical prior, localised, with a copy control (E22) |
| CoT faithfulness from outside | Turpin 2023; Lanham 2023; Chen 2025; Tutek 2025 | measured inside the model under a known cue |
| Misleading names in code | Le et al. 2026; Le et al. 2025; CodeCrash | localisation in controlled arithmetic, not accuracy on real code |
| Binding | Feng & Steinhardt 2024; Wu et al. 2025; Gur-Arieh et al. 2026 | binding versus the name's own meaning; "lexical" used in our sense only |
| Distraction | Shi et al. 2023; GSM-Symbolic | the distractor is the variable's name |
| Design lineage | Stroop 1935; Geirhos et al. 2019 | one-to-one transfer, with a lure index computed their way |

## 10. Consequences for the experiments

- **E22 (new):** rule out positional copying in lure errors at P5 (Liu 2026).
- **E3:** report a Geirhos-style lure index beside accuracy.
- **E8:** report Zhang–Nanda normalised logit difference beside recovery rate.
- **E6:** report the Hewitt–Liang ceiling with selectivity.

## 11. Citation flags before submission

Verify venue lines not printed in the PDFs on disk: Lin et al. 2025 (likely ACL 2025),
Prabhakar et al. 2024 (likely Findings of EMNLP 2024), Tutek et al. 2025 (EMNLP 2025 per
Anthology, PDF is arXiv v4), Stolfo 2023 (EMNLP 2023), Zhu 2025 (COLING 2025), Ye et al. 2025
(ICLR 2025; page numbers from arXiv v1), Hu et al. 2026 (arXiv id and version).
