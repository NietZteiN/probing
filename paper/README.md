# `paper/`

ACL/ARR **short paper** (4 pages of body; references, Limitations and appendix excluded).
`\usepackage[review]{acl}` = anonymous with line numbers, as ARR wants at submission.

- `main.tex` — the draft. Prose exists for everything that is argument rather than result;
  every quantity is `\NUM{key}`, rendered red until `scripts/51_tables.py` writes `numbers.tex`.
  The Results section is written as the shape of each paragraph so any outcome cell of
  PLAN.md §5 can be filled without restructuring.
- `refs.bib` — verification status per entry in `../papers/REFERENCES.md`.
- `acl.sty`, `acl_natbib.bst` — official ACL style files, committed so the draft builds from a
  clean checkout.
- `tables/`, `figures/` — generated; never hand-edited.

```bash
make paper      # tectonic build + body page count
make numbers    # every unfilled \NUM{} key with its count
```
