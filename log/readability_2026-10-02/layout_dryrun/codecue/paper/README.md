# ACL ARR LaTeX folder

**Misleading Identifiers Change What Code Traces Write** — an ARR **short paper**, with a **4-page main-content limit**.

`main.tex` uses the standard 11pt article class and `\usepackage[review]{acl}`:
anonymous authors, line numbers, page numbers, A4 paper and two columns throughout,
including appendices. Wide figures and tables use full-width floats. Limitations follow
the conclusion, before references; appendices follow references. The abstract is limited
to 200 words. No template margins, fonts or vertical spacing are overridden.

The requirements were checked against the [ARR call for papers](https://aclrollingreview.org/cfp)
and [ACL formatting guidelines](https://acl-org.github.io/ACLPUB/formatting.html) on 2026-10-02.
`acl.sty` and `acl_natbib.bst` match the [official ACL files](https://github.com/acl-org/acl-style-files)
byte for byte; their SHA-256 hashes are recorded in `check_arr.py`.

## Files

- `main.tex`, `main.pdf` — manuscript and compiled review draft.
- `acl.sty`, `acl_natbib.bst`, `refs.bib` — local style and bibliography dependencies.
- `numbers.tex`, `tables/`, `figures/` — generated results and assets; update them through
  their generators rather than editing measured values by hand.
- `Makefile`, `check_arr.py`, `page_limit.txt` — standalone build and checks.


## Build and check

From the repository root, using the existing cluster environment:

```bash
make paper             # compile and check formatting; errors propagate
make paper-check       # check the existing PDF
make paper-submission  # also reject unfilled numbers, missing assets and unresolved references
```

The folder also builds independently. With Tectonic and Python with `pypdf` installed:

```bash
cd paper
make TECTONIC=/path/to/tectonic PY=/path/to/python
make submission TECTONIC=/path/to/tectonic PY=/path/to/python
```

For Overleaf, upload the manuscript, local style/bibliography files, `numbers.tex`,
`tables/` and `figures/`, and select `main.tex` as the main document. All manuscript
inputs live inside this folder.

The automated checks verify the official styles, review mode, section order, page budget,
abstract length, A4 portrait pages and embedded fonts, including fonts in figures.
They report draft placeholders separately; `--strict` makes these a failure. Visual
legibility, citation accuracy and scientific completeness still require review.
