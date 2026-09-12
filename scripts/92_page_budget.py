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
        if i and (re.search(r"(?m)^\s*Limitations\s", t) or re.search(r"(?m)^\s*References\s", t)):
            body_end = i + 1; break
    tex = (ROOT / "paper" / "main.tex").read_text()
    tex = re.sub(r"(?m)%.*$", "", tex)
    nums = sorted(set(re.findall(r"\\NUM\{([^}]*)\}", tex)))
    print(f"pages: {len(pages)} total, body ends on page {body_end} (limit {LIMIT}); unfilled numbers: {len(nums)}")
    return 0 if body_end <= LIMIT else 1


if __name__ == "__main__":
    sys.exit(main())
