# Paper Plan. Cue Conflict in Chain-of-Thought Arithmetic

Target venue is the October 2026 ARR cycle (NAACL 2027 or COLING 2027), submitted as a short paper. The ARR deadline is October 12, 2026 (AoE). Every author must also register as a reviewer by that date. Findings versus main track is decided by the conference at commitment, not at submission.

---

## 1. The Core Idea

Misleading variable names give us a cue-conflict test of whether chain of thought actually computes.

Kudo et al. (EACL Findings 2026) showed that intermediate values become linearly decodable during chain of thought in controlled multi-step arithmetic. In their clean tasks, variable names are either meaningless letters or agree with the values. Under those conditions a probe cannot tell whether the model computed a value or picked it up from a surface cue. Placing the name's meaning in conflict with the true value separates the two explanations.

This is the same logic as the Stroop task in cognitive science, where the word RED printed in blue ink reveals whether reading overrides color naming. It is also the logic of shape versus texture conflict images in vision (Geirhos et al., 2019), which showed that apparent shape recognition in CNNs was partly texture matching. The paper brings this established method to arithmetic reasoning in language models.

### One-sentence thesis

> When a variable's name contradicts its value, does chain of thought let the model's computation override the name, and where inside the model does that happen?

Every section, figure, and experiment should serve this sentence. Anything that does not goes to the appendix or to future work.

### Title options

1. *Name or Value? Cue Conflict Reveals When Chain of Thought Overrides Lexical Priors*
2. *Does Chain of Thought Compute or Read? A Stroop Test for Multi-Step Arithmetic*
3. *When the Name Lies. Probing Lexical Interference During Chain-of-Thought Arithmetic*

Option 1 is the safest for reviewers because it names the method and the finding space. Option 2 is the most memorable.

---

## 2. Motivation and Gap

The argument has three steps.

1. **Names carry meaning in code and math, and misleading names hurt models.** Behavioral work on code shows accuracy drops when identifiers imply the wrong behavior (When Names Disappear, 2025) and when misleading natural-language cues are embedded in code (CodeCrash, 2025). A prior human and model comprehension study found that adversarial renaming collapsed model accuracy to 21% with a 10% rate of high-confidence errors. Cite this in the third person for anonymity.
2. **Behavioral drops cannot say where the interference happens.** An accuracy drop is compatible with a corrupted input representation, a corrupted computation, or a late readout error. These have different implications for faithfulness and for fixing the problem.
3. **Interpretability work on clean tasks cannot say what the decoded value reflects.** Kudo et al. show when values appear, but in clean tasks the computed value and any surface cue never disagree.

The gap is therefore a controlled setting where the cue and the computation disagree, observed from the inside.

---

## 3. Research Questions

**RQ1. Behavior.** Do incongruent names reduce accuracy relative to neutral names, and do congruent names increase it? How do these effects change with and without chain of thought?

**RQ2. Representation.** Beyond the name token itself, at which layers and positions is the lure value decodable? At which reasoning step does the true value overtake the lure?

**RQ3. Causation.** Does removing the name's influence, by patching in activations from the matched neutral instance, restore correct answers?

Model breadth is not its own question. It appears as a robustness check across RQ1 to RQ3.

---

## 4. Experimental Design

### 4.1 Task format

Reuse Kudo et al.'s generator and difficulty levels so results are directly comparable. Their code and data are public. The running example in the paper should be a two-step problem.

```
Neutral       pen=1+cup, cup=2+3; pen=?
Congruent     six=1+five, five=2+3; six=?
Incongruent   pen=1+two, two=2+3; pen=?
```

In the incongruent example, `two` has true value 5 and lure value 2. The queried variable `pen` has true value 6.

### 4.2 Conditions

| Condition | Name of the target variable | Purpose |
|---|---|---|
| Letter baseline | Single letter (A, B) | Connects to Kudo et al.'s original format |
| Neutral | Single-token non-number word | Main comparison point |
| Congruent | Number word equal to the true value | Measures facilitation |
| Incongruent | Number word not equal to the true value | Measures interference |
| Irrelevant lure (control) | Number word naming a distractor variable outside the queried chain | Separates lexical presence of a number from a binding conflict |

### 4.3 Generation rules

These rules prevent the labeling problem in the recruiting flyer example, where `four=1+two` made both names misleading.

1. Exactly one variable per instance carries a number-word name. All other variables use neutral words.
2. The lure value differs from the true value and from every intermediate value in the problem, so the lure is never correct for another reason.
3. The lure value falls inside the probe's class range, so the probe can express it.
4. Neutral words are matched to number words in token count under each model's tokenizer. This keeps positions aligned for patching and removes a length confound. Check tokenization separately for every model family.
5. Vary the position of the misleading variable in the chain (queried variable versus intermediate variable) and record the distance between lure and true value as a covariate.
6. Each incongruent instance has a matched neutral and congruent twin with identical arithmetic. All comparisons are within these matched sets.

### 4.4 Prompting regimes

- **Chain of thought.** The model writes each step, for example `cup=2+3=5, pen=1+5=6`. Note that the chain of thought writes the name next to the computed value, which may help the model recover.
- **Direct answer.** The model outputs the answer with no intermediate steps.

The contrast between these two regimes is the center of the paper.

### 4.5 Models

Start with two models that Kudo et al.'s code already supports. Then add families up to three or four, at two sizes each where possible. Include at least one base versus reasoning-tuned pair. Reasonable candidates are Qwen, Llama, Gemma, and OLMo models in the 1B to 8B range. Final choices depend on compute and tokenizer behavior.

### 4.6 Probe positions

Using the incongruent example, probe at these positions.

| Label | Position | Role in the argument |
|---|---|---|
| P1 | The name token `two` at its definition | Expected to decode the lure trivially. Reported only as a reference, never as evidence |
| P2 | The `=` after the definition | First position where the true value can be computed |
| P3 | The query position before any chain of thought | Tests whether the answer is set before reasoning |
| P4 | Chain-of-thought tokens where the variable's value is written | Tests recovery during reasoning |
| P5 | The final answer position | Links internal state to output |

All claims about lure interference are made at P2 to P5.

### 4.7 Probe training

Train probes on the neutral condition only, then evaluate them on congruent and incongruent instances. A probe trained this way reads out the model's value representation. If it outputs the lure on incongruent inputs, the representation itself has shifted toward the lure. Training on incongruent data would let the probe learn the lure directly and would weaken this argument.

Report selectivity against a control task (Hewitt and Liang, 2019) so that high accuracy cannot be attributed to probe capacity alone. Use several random seeds and report the spread.

### 4.8 Metrics

**Behavior**

- Accuracy per condition.
- Lure rate, the share of answers equal to the lure value. This separates lure errors from other errors.
- Interference, defined as neutral accuracy minus incongruent accuracy.
- Facilitation, defined as congruent accuracy minus neutral accuracy.

**Representation**

- True-value probe accuracy per layer and position.
- Lure mass, the probe probability assigned to the lure class.
- Margin, defined as log p(true) minus log p(lure). The crossover point is the first reasoning step where the margin becomes positive and stays positive.

**Instance-level link**

- Whether the margin at P2 to P4 predicts behavioral lure errors, using logistic regression with model and difficulty level as covariates.

**Statistics**

- Bootstrap confidence intervals over instances.
- Mixed-effects logistic regression for behavior, with random effects for problem template.
- FDR correction across layer and position tests.

### 4.9 Causal test

Patch residual stream activations at the misleading name's token positions from the matched neutral run into the incongruent run. Sweep layers. Measure the recovery rate, the share of lure errors that become correct after patching. As a control, patch in activations from a different neutral word to confirm that recovery is not caused by patching itself. Reuse Kudo et al.'s patching setup where possible, following standard practice for activation patching (Zhang and Nanda, 2024).

An optional second test is to remove the lure direction learned by a lure-specific probe and measure the effect on accuracy.

---

## 5. Making Every Outcome Informative

The introduction can be drafted before results arrive because every combination of internal and behavioral results says something about chain-of-thought faithfulness.

| | Model answers correctly | Model answers with the lure |
|---|---|---|
| **Lure fades inside the model** | Computation overrides the name. This extends Kudo et al.'s claim to conflicting input | The error enters late, at readout. A new failure location |
| **Lure persists inside the model** | Hidden interference. The output is correct but the internal state is contaminated | Faithful computation from a corrupted value. The chain of thought is faithful but wrong |

The chain-of-thought versus direct-answer comparison then shows whether writing out the steps moves the model between cells. The most likely publishable headline has the form "chain of thought reduces but does not remove lexical interference, and the true value overtakes the lure at step K."

---

## 6. Positioning

### Relation to Kudo et al.

Present this as building on their work, not correcting it. They established when answers are computed. This paper asks what that computation is sensitive to. Credit the public code explicitly. Sakaguchi is a coauthor of the original paper, so a short courtesy note to the Tohoku group before submission is worthwhile, and it also checks for overlap with any follow-up of their own.

### Related work map

| Area | Representative work | How this paper differs |
|---|---|---|
| Chain-of-thought internals | Kudo et al. (2026) | Adds cue conflict to test what the decoded signal reflects |
| Chain-of-thought faithfulness | Turpin et al. (2023), Lanham et al. (2023) | Measures faithfulness inside the model under a controlled conflicting cue |
| Misleading names in code | When Names Disappear (2025), CodeCrash (2025), prior renaming study (third person) | Controlled arithmetic and localization, rather than accuracy drops on real code |
| Variable and entity binding | Feng and Steinhardt (2024), Wu, Geiger, and Millière (2025) | Tests whether a value binds to a name independent of the name's meaning |
| Distraction in arithmetic | Shi et al. (2023) | The distractor is the variable's own name, not extra context |
| Cue-conflict methodology | Stroop (1935), Geirhos et al. (2019) | Transfers the design to language model reasoning |

Verify every citation, year, and venue before submission.

---

## 7. Wording and Scope Decisions

1. **Say "misleading" or "incongruent," not "adversarial."** The names are handcrafted, not optimized. "Adversarial" invites the question of why there was no search for worst-case lures.
2. **Keep claims about arithmetic with variable assignment.** Code comprehension is the motivation, not the finding. Overclaiming toward code invites requests for real programs.
3. **Never present P1 results as evidence.** Say explicitly that decoding the lure at the name token is expected, which shows reviewers the confound was anticipated.
4. **Use "interference" and "facilitation" consistently.** Borrowing Stroop vocabulary signals the design lineage without extra explanation.

---

## 8. Introduction Outline

**Paragraph 1. The problem.** Models read code and math where names carry meaning. Misleading names reduce accuracy, as recent code benchmarks and a prior comprehension study show. Accuracy drops cannot show where inside the model the name interferes.

**Paragraph 2. The limitation of current interpretability.** Probing and patching studies show that values are computed during chain of thought on clean tasks. In clean tasks the cue and the computation never disagree, so the decoded value could come from either.

**Paragraph 3. The method.** Cue conflict resolves this, as in the Stroop task and in texture versus shape studies of vision models. Introduce congruent, neutral, and incongruent versions of multi-step arithmetic with matched twins, and the chain-of-thought versus direct-answer contrast.

**Paragraph 4. Findings and contributions.** State the main finding in one sentence, placed in the correct cell of the outcome table. Then list the contributions.

- A cue-conflict extension of a controlled arithmetic benchmark with matched instances.
- A layer, position, and reasoning-step map of lure versus true value, with and without chain of thought.
- Causal evidence from activation patching on whether the name's influence explains the errors.

---

## 9. Short Paper Structure

ARR short papers allow four pages of content, plus references, a required limitations section, and an appendix.

| Section | Length | Content |
|---|---|---|
| Introduction | 0.75 page | Paragraphs from Section 8, with Figure 1 |
| Setup | 0.75 page | Conditions, generation rules, probe positions, models |
| Results | 1.75 pages | RQ1 behavior, RQ2 crossover map, RQ3 patching |
| Related work | 0.5 page | Condensed version of the Section 6 table |
| Conclusion | 0.25 page | One finding, one implication for faithfulness |
| Limitations | Outside page limit | Small models, synthetic tasks, handcrafted names, English number words only |
| Appendix | Outside page limit | Full layer sweeps, tokenizer checks, extra models, selectivity controls |

### Figures and tables

1. **Figure 1.** The three matched versions of one problem, color coded, with a sketch of the probe margin across reasoning steps. This figure should carry the whole idea on its own.
2. **Table 1.** Accuracy, lure rate, interference, and facilitation per model and regime.
3. **Figure 2.** Heatmap of probe margin over layers and positions P2 to P5, incongruent condition, chain of thought versus direct answer side by side.
4. **Figure 3.** Patching recovery rate by layer, with the control patch as a baseline.

---

## 10. Anticipated Reviewer Questions

| Question | Answer in the paper |
|---|---|
| Is this just a robustness check of Kudo et al.? | No. Cue conflict tests what the decoded signal reflects, which clean tasks cannot do. The chain-of-thought contrast tests the scope of their central claim |
| Won't a probe trivially decode "two" as 2? | Yes at P1, which is why P1 is excluded from evidence. Probes are trained on neutral data only and lure claims are made at P2 to P5 |
| Why arithmetic instead of real code? | Control. Matched twins with identical arithmetic are impossible to build at scale in real code. Claims are scoped accordingly |
| Why these names and not optimized attacks? | The question is whether meaningful names interfere, not how badly a search can break the model. Optimized lures are future work |
| Probing is correlational | RQ3 provides patching evidence with a control patch |
| Could tokenization explain the effect? | Neutral words are matched in token count per tokenizer, reported in the appendix |
| Only small models | Compute limits are stated. Within-family size comparisons show the trend direction |

---

## 11. Abstract Template

Fill in brackets once results exist.

> Language models are known to be misled by variable names that contradict a program's behavior, but accuracy drops cannot show where inside the model the name interferes. Interpretability studies find that intermediate values are computed during chain of thought, yet in their clean tasks the variable name and the computed value never disagree. We introduce a cue-conflict test for multi-step arithmetic in which each problem appears with congruent, neutral, and incongruent variable names under identical arithmetic. Across [N] models, incongruent names reduce accuracy by [X] points without chain of thought and by [Y] points with it. Linear probes trained on neutral problems show that the lure value remains decodable [until step K / through the final answer], and activation patching from matched neutral problems recovers [Z]% of lure errors. These results indicate that chain of thought [overrides / reduces but does not remove] lexical interference, and they [support / qualify] the view that chain-of-thought computation is faithful.

---

## 12. Timeline to October 12

| Dates | Goal | Deliverable |
|---|---|---|
| Sep 11 to Sep 17 | Reproduce Kudo et al.'s probing result on one model. Build the generator with all conditions and matched twins. Run tokenizer checks | Reproduced baseline figure, validated dataset |
| Sep 18 to Sep 24 | Behavioral runs and hidden-state caching on two models, both regimes. First probe sweeps | Table 1 draft, first heatmap |
| Sep 25 to Oct 1 | Crossover analysis and patching on two models | Figures 2 and 3 drafts |
| **Oct 1** | **Go or no-go.** Submit if the core effect is clean on two models. Otherwise move to the next ARR cycle | Decision |
| Oct 2 to Oct 8 | Third model if time allows. Write the full draft | Complete draft |
| Oct 9 to Oct 12 | Anonymize, limitations section, responsible NLP checklist, reviewer registration for all authors | Submission |

A rushed submission carries a cost. ARR resubmissions are linked to their earlier reviews, so a weak first round follows the paper.

---

## 13. Team Task Split

This follows the four steps on the recruiting flyer.

| Role | Task | Output |
|---|---|---|
| Puzzle builder | Extend the generator with conditions, generation rules, and matched twins. Write unit tests for rules 1 to 3 in Section 4.3 | Dataset and tests |
| Model runner | Run both regimes, record answers, cache hidden states at P1 to P5 | Behavioral results and activation files |
| Probe trainer | Train neutral-only probes per layer and position, compute margins, run selectivity controls | Probe results and heatmaps |
| Analyst | Behavioral statistics, crossover analysis, instance-level regression | Table 1 and statistics |
| Lead (Jack) | Patching experiments, framing, writing, integration | Figure 3 and the paper |

The patching experiment should stay with the lead because it is the part reviewers will scrutinize most and the part most likely to break on position alignment.

---

## 14. Limitations to State Up Front

- Synthetic arithmetic with short chains, so results may not transfer directly to real code.
- Models up to roughly 8B parameters.
- Handcrafted number-word names in English only. Digit-based names (such as `x2`) and other languages are future work.
- Linear probes can miss nonlinearly encoded information. Patching partly addresses this.
