# Chain of Thought Limits Lure Answers in Arithmetic

Updated 2026-10-03 to match the results and revised manuscript. This replaces the earlier,
prospective outline; the preregistration and its amendments remain the record of the design.

The paper asks whether writing a chain prevents a misleading variable name from determining
the answer. Its central finding is that chains limit lure answers in almost every tested
contrast. The readout comparison is descriptive and sensitive to overall task competence and panel
composition. The completed comparison covers nine models and eighteen model-variable pairs;
removing OLMo-2-1B reverses the model-mean correlations. Generated-chain
transfer is strong for Llama-3.2-3B but fails the neutral calibration threshold for OLMo-2-1B.
Training directly on generated neutral chains improves OLMo's calibration but still fails
in all six held-out cells; Llama passes all six with the same design.
These readouts do not establish why chains reduce lure answers.

## Order of the evidence

1. **Show the conflict.** One arithmetic problem has a correct answer and a distinct digit
   implied by the name. Changing the name preserves the arithmetic.
2. **Establish the behavioral contrast.** Direct answers show naming effects. Most chain
   contrasts meet the lure-equivalence bound, although accuracy effects can remain.
3. **Compare readouts and competence.** Compare the true-value and lure readouts
   at the writing step. Report the unresolved instance-level regression and the matched
   written-value control alongside this comparison. Include within-model role differences,
   leave-one-model-out sensitivity, gold-trained transfer and generated-neutral training
   with computation-disjoint validation and testing.
4. **Ask what patching localizes.** Present the early-layer sensitivity in Llama-3.2-3B,
   with disruption and alternative-lure controls. Other models do not establish a common
   name-specific mechanism.

The main text keeps these results, the running example and the methods needed to interpret
them. The appendix holds task levels, replication, equivalence details, tokenization,
additional naming controls, full probe sweeps, tuning comparisons and alternative chains.

## Limits that affect the argument

- Lure equivalence concerns numeric lure answers, not the absence of all naming effects.
- A readout at a name token or query can reflect name identity; it is not evidence of a
  computed value. The selectivity control must accompany that interpretation.
- Most probes and patches use a force-decoded gold chain. Generated-chain behavior is a
  separate measurement, and its error counts cannot replace the patching denominators.
- Generated-chain probes fail neutral calibration for OLMo-2-1B even when trained on
  generated neutral chains. Llama-3.2-3B passes,
  but its lure-writing cases are too rare to establish a general mechanism.
- Model-mean correlations are descriptive. The two variable roles are not independent
  model replications, and the correlation direction depends on OLMo-2-1B.
- A correct value stated in a generated chain does not alone explain the behavioral reduction.
- A patch replaces a mixed representation. Recovery with substantial disruption, or similar
  removal by another lure, cannot establish a clean name-specific causal pathway.
- Arithmetic and code are separate studies. Their different readouts do not establish a
  shared mechanism.

## Writing conventions

Start with the problem and follow it through the results. Use headings that state findings,
short paragraphs and concrete verbs. Avoid contribution lists, claim ladders and repeated
statements of scope. Keep measured quantities in generated `\NUM{}` macros, report interval
conventions explicitly, and preserve inconclusive or contrary results.
