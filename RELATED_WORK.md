# Related work, and what is left for this paper to claim

*Written 2026-09-12. Kudo et al. was read in full (`papers/kudo2026faithful.pdf`); the others
were verified by title, authors, venue and year against arXiv/OpenReview/ICML pages on
2026-09-12 and are listed in `papers/REFERENCES.md` with what was checked. Any claim below
about a paper's content that goes beyond its abstract is marked "(abstract)" and must be
confirmed from the PDF before it appears in the paper.*

## 1. The paper this one builds on

**Kudo, Aoki, Kuribayashi, Sone, Taniguchi, Brassard, Sakaguchi, Inui (2026). LLMs Faithfully
and Iteratively Compute Answers During CoT: A Systematic Analysis With Multi-step Arithmetics.
Findings of EACL 2026, pp. 1114–1153.** arXiv:2412.01113 (earlier title: *Think-to-talk or
talk-to-think?*). Code: github.com/keitokudo/faithful-cot-multistep-arithmetic.

What they show (read, §3–4): with linear probes per (position, layer, variable) on synthetic
`A=1+B, B=2+3; A=?` problems at five levels, every sub-answer that needs at least one
computation step becomes linearly decodable only **after** the chain begins (t*_eq > 0, τ = 0.9;
Table 2, Table 3 across nine models incl. Llama-3.2-3B and Llama-3.1-8B). Pre-CoT accuracy
peaks around 60% in the second half of the defining equation. Activation patching in a
(equation × 4-layer) grid shows the final answer depends causally on the chain, barely on the
input (Fig. 5), with strong recency bias (Fig. 6). Probes: single linear layer, SGD lr 1e-3,
batch 10,000, 10,000 epochs (Table 8). Task accuracy ≈ 100% for all models except
Llama-3.2-3B at levels 3–5 (93.2 / 90.9 / 38.5; Table 7). Implicit (no-chain) format: 77.8%
accuracy and no position reaches τ (Table 6).

**What they cannot say, and this paper adds.** In every one of their instances the variable
name is a letter, so the decoded value has exactly one possible source. Nothing in their
design can tell "the model computed 5" from "the model read 5 off a cue" because no cue ever
disagrees. Our congruent/incongruent twins create the disagreement; the probe (trained on
neutral names) then reads which of the two the representation holds, and patching says whether
the name causes the error. Their central claim — sub-answers are derived during CoT — is
extended, not challenged: we ask what that derivation is sensitive to.

Their format and probe recipe are reproduced exactly (Level 3, three same-level shots, the same
probe hyperparameters) so our neutral-condition heatmap should reproduce their Figure 2 for
Llama-3.2-3B before anything else is claimed (docs/EXPERIMENT_PLAN.md §0).

## 2. Misleading names in code (the motivation)

**Le, Pham, Van, Phan, Phan, Nguyen (2025). When Names Disappear: Revealing What LLMs
Actually Understand About Code.** arXiv:2510.03178. (abstract) Semantics-preserving
obfuscations remove the naming channel; summarisation collapses to line-by-line description and
execution tasks that should depend only on structure also drop; releases ClassEval-Obf.

**Lam et al. (2025). CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code
Reasoning.** NeurIPS 2025. arXiv:2504.14119. (abstract) Perturbs CRUXEval and LiveCodeBench
with misleading comments/hints; 17 LLMs; average −23.2% on output prediction; models shortcut
by over-relying on NL cues.

**Prior renaming study (cite in the third person).** Adversarial renaming collapsed model
accuracy to 21% with a 10% rate of high-confidence errors. *Citation to be supplied by the
lead; the number is quoted from PLAN.md §2.*

Difference: all three are behavioural on real code. They establish that names interfere; they
cannot locate the interference. We give up realism for control: matched twins, identical
arithmetic, a probe-readable value space.

## 3. Chain-of-thought faithfulness

**Turpin, Michael, Perez, Bowman (2023). Language Models Don't Always Say What They Think.**
NeurIPS 2023. Biasing features in the prompt change answers without being mentioned in the
CoT. **Lanham et al. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning.**
arXiv:2307.13702. Truncation/corruption of the chain; larger models less faithful.
**Chen et al. (2025). Reasoning models don't always say what they think.** arXiv:2505.05410
(cited by Kudo). Difference: these measure faithfulness from the outside (does the answer track
the stated reason?). We measure it inside the model under a *controlled* conflicting cue, and
the outcome table (PLAN.md §5) maps each internal/behavioural combination onto a faithfulness
reading. Kudo et al. §5 already positions against Paul et al. (2024) and Bentham et al. (2024);
we inherit that positioning.

## 4. Variable and entity binding

**Feng & Steinhardt (2024). How do Language Models Bind Entities in Context?** ICLR 2024.
arXiv:2310.17191. Binding-ID mechanism in Pythia/LLaMA. **Wu, Geiger, Millière (2025). How Do
Transformers Learn Variable Binding in Symbolic Programs?** ICML 2025. arXiv:2505.20896.
(abstract) Training-dynamics phases for dereferencing assignment chains in a synthetic task.
Also relevant: **Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context**
(ICLR 2026, arXiv:2510.06182). Difference: binding work asks *how* a value attaches to a name;
we ask whether the attachment survives when the name itself carries a competing value.

## 5. Distraction in arithmetic

**Shi et al. (2023). Large Language Models Can Be Easily Distracted by Irrelevant Context.**
ICML 2023. Irrelevant sentences added to GSM problems. Difference: our distractor is the
variable's own name, not extra context; the level-4 `irrelevant` condition is exactly the
bridge (a number word present but not bound to the queried chain).

## 6. Cue-conflict methodology

**Stroop (1935)**, J. Exp. Psychol. 18(6):643–662. **Geirhos et al. (2019). ImageNet-trained
CNNs are biased towards texture.** ICLR 2019. Cue-conflict images separated shape from texture
reliance. **Hewitt & Liang (2019). Designing and Interpreting Probes with Control Tasks.**
EMNLP 2019: the selectivity control we report with every probe. **Zhang & Nanda (2024).
Towards Best Practices of Activation Patching.** ICLR 2024: the patching protocol Kudo et al.
and we follow.

## 7. Reviewer-facing summary (PLAN.md §6 table, condensed)

| area | representative | this paper differs by |
|---|---|---|
| CoT internals | Kudo et al. 2026 | adding cue conflict to say what the decoded value reflects |
| CoT faithfulness | Turpin 2023; Lanham 2023 | measuring inside the model under a controlled cue |
| misleading names in code | When Names Disappear; CodeCrash | localisation in controlled arithmetic, not accuracy on real code |
| binding | Feng & Steinhardt 2024; Wu et al. 2025 | testing whether binding is independent of the name's meaning |
| distraction | Shi et al. 2023 | the distractor is the name itself |
| cue conflict | Stroop 1935; Geirhos 2019 | transferring the design to LM reasoning |
