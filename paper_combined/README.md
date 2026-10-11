# Combined long-paper draft

**Misleading Names Can Distort Written Reasoning**

This is the consolidated alternative to the two short papers. The argument follows the
provided outline, with eight pages including Limitations. References and supporting
appendices follow in the same PDF.

The story starts with Kudo et al.'s evidence of computation during arithmetic reasoning.
We introduce names that suggest distinct wrong values, show when written reasoning
follows those names, recover correct digits immediately before wrong code writes, and
test demonstrations that show each source operation alongside its result.

The main paper follows seven sections:

1. Introduction
2. Experimental design
3. Misleading names can affect written reasoning
4. Correct values can precede incorrect writes
5. Showing the operations reduces errors
6. Related work
7. Discussion and conclusion

The abstract uses the supplied wording as its starting point. Three shared figures cover
the naming conflict, behavioral effects and readout timing; one table compares output
formats. Captions retain only essential information.

`main.pdf` is the anonymous ACL review draft. `submission_source.zip` contains its required
LaTeX sources, tables and figures, with no dependency on the neighboring repositories.
`abstract.txt` contains the plain-text abstract. No OpenReview submission has been made.

To build from this directory, run `make submission` with Tectonic and a Python environment
containing pypdf. To refresh imported evidence in the research workspace, run
`python scripts/91_combined_paper.py` from the probing repository, rebuild, then run the
same script with `--bundle`. The source script checks numerical keys and references;
`evidence_provenance.json` records the source summary hashes.

The arithmetic and code cohorts remain distinct. The draft does not pool their rates or
count overlapping checkpoints as independent model replications. The original code
selection deviation, independent-name replication, probe calibration failures and
competence changes remain explicit.
