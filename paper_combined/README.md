# Combined long-paper draft

**Correct Values Can Be Recovered Before Wrong Reasoning Steps**

This is the consolidated alternative to the two short papers, within the eight-page
limit. The current main text and Limitations occupy seven pages. References and supporting appendices follow in the same PDF.

The story starts with the gap between recovering a correct value and writing it.
Building on Kudo et al.'s controlled arithmetic task and methods, we introduce names
that suggest distinct wrong values. Code supplies the strongest three-part result:
misleading intermediate writes, accurate correct-digit readouts immediately before
those writes, and fewer errors under examples that show the source operation.
Arithmetic provides the foundation and comparison.

The main paper follows eight sections:

1. Introduction
2. Experimental design
3. Misleading names can change written steps
4. Correct values are recoverable before wrong steps
5. Showing the operation reduces written errors
6. Scope of the findings
7. Related work
8. Discussion and conclusion

The abstract presents the controlled naming manipulation, recovery on incorrect
outputs in three code models, and the final-answer improvement from 37.1% to 92.6%
in OLMo-2-7B.
A verified full-program example shows actual wrong and correct outputs under the two
example formats. Four main figures cover that example, readout timing with
exact recovery percentages and unique error-producing program counts, plainly labeled format controls, and fresh-name
replication with CodeGemma. Table 1 defines scoring; Table 2 reports correct first values
and final answers on the same misleading-name programs. The duplicated opening summary
table is removed, and the arithmetic/code behavior plot is now supporting material.

The methods introduce inputs, requested outputs, the broader model panels and sample
sizes, matched names, grading and measurement before uncertainty. Section 6 explains
accuracy confounds and the natural-code check in plain language; the discussion
distinguishes internal recoverability, intermediate correctness and final correctness.
Appendix A supplies prompt assembly, complete code demonstration
set 7, parsing and fitting details. Arithmetic is a compact comparison, with transfer
analyses and competence checks in the appendix. Related work distinguishes the result
from Orgad et al.'s answer-level findings, and Limitations explicitly allows probes to
recover input cardinality or related features. Figure labels use plain language;
the nearby prose explains the plotted positions, colors, error measure and meaning
of zero. Captions retain essential information.

`main.pdf` is the anonymous ACL review draft. `submission_source.zip` contains its required
LaTeX sources, tables and figures, with no dependency on the neighboring repositories.
`abstract.txt` contains the plain-text abstract. No OpenReview submission has been made.

To build from this directory, run `make submission` with Tectonic and a Python environment
containing pypdf. To refresh imported evidence in the research workspace, run
`python scripts/91_combined_paper.py` from the probing repository, rebuild, then run the
same script with `--bundle`. The source script checks numerical keys and references;
`evidence_provenance.json` records the source summary hashes. The refresh also runs
`scripts/93_combined_correctness.py`, a CPU-only audit of saved generations for the same
285-program format cohort. `format_correctness.json` retains counts, intervals and
portable source-file hashes. This analysis loads no models or tokenizers. `scripts/94_combined_example.py` verifies
the displayed generations against three original source files, and
`scripts/95_combined_demonstrations.py` typesets the actual example prompts.
`refs_extra.bib` preserves references added for the combined manuscript on refresh.

The arithmetic and code cohorts remain distinct. The draft does not pool their rates or
count overlapping checkpoints as independent model replications. The original code
selection deviation, independent-name replication, probe calibration failures and
competence changes remain explicit. Reported direct-answer and trace contrasts rename
different variables and do not isolate the effect of requesting a trace.
