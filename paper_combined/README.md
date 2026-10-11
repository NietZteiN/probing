# Combined long-paper draft

**Correct Values Can Be Recovered Before Wrong Reasoning Steps**

This is the consolidated alternative to the two short papers, with eight pages
including Limitations. References and supporting appendices follow in the same PDF.

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

The abstract presents the failure, error-conditioned recovery and demonstration change.
An opening table places those three results together. A second table reports correct
first writes and final answers on the same misleading-name programs. Five main figures
cover the naming conflict, behavioral effects, readout timing, operation controls and
fresh-name replication. The full format and correctness tables are in the appendix.
Captions retain only essential information.

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
portable source-file hashes. This analysis loads no models or tokenizers.

The arithmetic and code cohorts remain distinct. The draft does not pool their rates or
count overlapping checkpoints as independent model replications. The original code
selection deviation, independent-name replication, probe calibration failures and
competence changes remain explicit. Reported direct-answer and trace contrasts rename
different variables and do not isolate the effect of requesting a trace.
