#!/usr/bin/env python
"""ARR SHORT paper: 4 pages of body; references, Limitations and appendix excluded. Prints the
body page count from paper/main.pdf and the number of unfilled \\NUM{} placeholders."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 4


def main() -> int:
    pdf = ROOT / "paper" / "main.pdf"
    if not pdf.exists():
        print("paper/main.pdf not built", file=sys.stderr); return 1
    from pypdf import PdfReader
    pages = [p.extract_text() or "" for p in PdfReader(str(pdf)).pages]
    body_end = len(pages)
    # review mode appends line numbers, so the heading is not alone on its line: match the
    # heading word followed by whitespace, on any page after the first.
    for i, t in enumerate(pages):
        # review mode glues the line number to the heading ("Limitations278"), so no \b here
        m = re.search(r"(?m)^\s*(Limitations|References)", t) if i else None
        if m:
            # the page still holds body text unless the heading is at its very top
            body_end = i + 1 if m.start() > 0.04 * len(t) else i
            break
    tex = re.sub(r"(?m)%.*$", "", (ROOT / "paper" / "main.tex").read_text())
    keys = set(re.findall(r"\\NUM\{([^}]*)\}", tex))
    nf = ROOT / "paper" / "numbers.tex"
    defined = set(re.findall(r"NUMval@([^\\]+)\\endcsname", nf.read_text())) if nf.exists() else set()
    unfilled = sorted(keys - defined)
    # a filled number still renders red if numbers.tex is stale, so also look for the marker in the PDF
    red = sum(t.count("\u27e8\u27e8") for t in pages)
    print(f"pages: {len(pages)} total, body ends on page {body_end} (limit {LIMIT}); "
          f"\\NUM keys {len(keys)}, defined {len(keys & defined)}, unfilled {len(unfilled)}"
          + (f" -> {unfilled}" if unfilled else ""))
    return 0 if body_end <= LIMIT else 1


if __name__ == "__main__":
    sys.exit(main())
