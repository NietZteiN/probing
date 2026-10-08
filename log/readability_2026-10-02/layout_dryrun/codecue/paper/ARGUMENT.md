# Misleading Identifiers Change What Code Traces Write

Updated 2026-10-02 to match the revised manuscript.

The paper follows one conflict: a variable is computed as a length but named like a sum.
A terse trace can write the sum. The revised probes train on disjoint programs and report readouts
and name controls under all three demonstration sets; decodability does not establish use. Showing the expression in demonstrations sharply reduces the effect.

## Order of the evidence

1. **Show an actual trace.** The same program receives different traces when worked examples
   show values alone or expressions with values.
2. **Measure the naming effect.** Establish written-lure excess against matched neutral
   twins and distinguish it from the models' ordinary length-step errors.
3. **Test sensitivity with patches.** Report name-site and decision-site interventions.
   Report the validated disjoint probe readouts and their controls, with demonstration-set
   repeats. Decodability does not establish that the model uses the decoded digit.
4. **Change the demonstrated format.** Hold the program and example identities fixed.
   Report the remaining OLMo effect, then use REPL, comments, scale and natural code to bound
   the finding.

The appendix holds task levels, identifier-meaning checks, the operation matrix, format
intervals, probe and patch details, scope checks and statistical conventions.

## Limits that affect the argument

- The original probes reused test programs in training. Reported replacement runs pass
  the split audit; retain the original caches as an audit trail.
- Decodability does not prove that computation is intact or that the model uses the decoded
  value. The name-identity controls have low selectivity.
- Patches replace mixed features. The alternative-lure control has no distinct source value
  in the main sum-family cell, and the affected Llama-3.1-8B name lacks aligned neutral twins.
- Expression-bearing examples reduce the effect; they do not eliminate it in every model.
- The CodeLlama-34B maximum-step interval crosses the equivalence margin. A small estimate
  does not establish equivalence.
- CRUXEval direct-answer and prose tests do not test the synthetic few-shot trace format.
- The companion arithmetic paper studies a different task. The two findings do not establish
  a shared causal mechanism.
- The planned panel meaning gate did not filter the runs. Report that deviation and avoid
  assuming each model interprets every retained name according to the operational name table.

## Writing conventions

Follow the problem through the evidence. Keep headings concrete, paragraphs short and
interpretations beside the measurements that support them. Avoid contribution lists and
repeated scope statements. Use generated `\NUM{}` macros for measured quantities and state
the interval convention actually used by the analysis.
