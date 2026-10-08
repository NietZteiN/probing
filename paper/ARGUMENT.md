# Chain of Thought Reduces Errors from Misleading Names in Arithmetic

Updated 2026-10-08 to match the results and revised manuscript. This replaces the earlier,
prospective outline; the preregistration and its amendments remain the record of the design.

Prior work follows arithmetic values during CoT. We add a competing digit in a variable's
name while preserving the equations, allowing a matched test of resistance to that word.
The implication is that writing a calculation can reduce this specific distraction:
robustness to a known conflicting cue becomes a concrete criterion for reasoning prompts.

The opening follows one problem: a variable named `four` has the computed value 6.
Answer-only prompting requests the final digit; CoT prompting requests the equations and
their evaluation before the answer. The paper asks whether this calculation reduces answers
based on the misleading name. Its central finding is that CoT limits name errors in almost every tested
contrast. The readout comparison is descriptive and sensitive to overall task competence and panel
composition. The completed comparison covers nine models and eighteen model-variable pairs;
removing OLMo-2-1B reverses the model-mean correlations. Generated-chain
transfer is strong for Llama-3.2-3B but fails the neutral calibration threshold for OLMo-2-1B.
Training directly on generated neutral chains improves OLMo's calibration but still fails
in all six held-out cells; Llama passes all six with the same design.
These readouts do not establish why chains reduce name errors. Keep this internal analysis
separate from the primary behavioral contribution. Overall accuracy changes and excess
answers matching the misleading digit are distinct outcomes; the equivalence finding
concerns the latter. The companion code paper instead tests readouts before actual wrong
writes and manipulates the demonstrated trace format.

Introduce linear probes through their purpose: predicting the correct digit from internal
activations. A probe recovering 6 and a language model answering 6 are separate observations.
End the introduction with the demonstrated reduction in answers matching misleading names
and the contribution of the paired renaming test. Report correlation sensitivity and
causal limits in Results and Limitations rather than making them the introduction's ending.

## Order of the evidence

1. **Show the conflict.** One arithmetic problem has a correct answer and a distinct digit
   implied by the name. Changing the name preserves the arithmetic.
2. **Establish the behavioral contrast.** Direct answers show naming effects. Most chain
   contrasts meet the name-error equivalence bound, although accuracy effects can remain.
3. **Compare readouts and competence.** Compare the true-value and readout of the name-suggested values
   at the writing step. Report the unresolved instance-level regression and the matched
   written-value control alongside this comparison. Include within-model role differences,
   leave-one-model-out sensitivity, gold-trained transfer and generated-neutral training
   with computation-disjoint validation and testing.
4. **Ask what patching localizes.** Present the early-layer sensitivity in Llama-3.2-3B,
   with disruption and alternative-name controls. Other models do not establish a common
   name-specific mechanism.

The main text keeps these results, the running example and the methods needed to interpret
them. The appendix holds task levels, replication, equivalence details, tokenization,
additional naming controls, full probe sweeps, tuning comparisons and alternative chains.

## Limits that affect the argument

- The name-error equivalence bound concerns answers suggested by misleading names, not the absence of all naming effects.
- A readout at a name token or query can reflect name identity; it is not evidence of a
  computed value. The selectivity control must accompany that interpretation.
- Most probes and patches use a force-decoded gold chain. Generated-chain behavior is a
  separate measurement, and its error counts cannot replace the patching denominators.
- Generated-chain probes fail neutral calibration for OLMo-2-1B even when trained on
  generated neutral chains. Llama-3.2-3B passes,
  but its name-error cases are too rare to establish a general mechanism.
- Model-mean correlations are descriptive. The two variable roles are not independent
  model replications, and the correlation direction depends on OLMo-2-1B.
- A correct value stated in a generated chain does not alone explain the behavioral reduction.
- A patch replaces a mixed representation. Recovery with substantial disruption, or similar
  removal by another name-suggested value, cannot establish a clean name-specific causal pathway.
- Arithmetic and code are separate studies. Their different readouts do not establish a
  shared mechanism.

## Writing conventions

Report main-text probe accuracy as percentages and selectivity differences as percentage
points. Distinguish states along a supplied correct calculation from states along the
model's own generated output. A neutral modal prediction is a visual reference; matching
it alone does not show that a state lacks information. Name-error equivalence concerns
the specific misleading digit, while overall accuracy can still change.

For readers outside this project, keep the abstract's result–evidence–implication order
but explain the task and comparison in ordinary words. Define the requested calculation
or reported variable values before using technical labels. Avoid unexplained equivalence,
matched-pair, activation and trace terminology; keep the central numbers and precise scope.

The abstract states the behavioral result first, summarizes the matched-name comparison
and equivalence result, then explains the implication for robustness in arithmetic
reasoning. Keep the running example in the introduction and secondary probe findings
in the body.

Start with the problem and follow it through the results. Use headings that state findings,
short paragraphs and concrete verbs. Avoid contribution lists, claim ladders and repeated
statements of scope. Keep measured quantities in generated `\NUM{}` macros, report interval
conventions explicitly, and preserve inconclusive or contrary results.

Terminology: a name-suggested value is the incorrect digit implied by the misleading name.
A name error writes that digit; excess name errors subtract the matched neutral twin's
rate of writing the same digit. Probe predictions of that value are a separate readout.
