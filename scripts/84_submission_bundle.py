#!/usr/bin/env python
"""Package only the LaTeX files and assets used by the submission manuscript."""
from pathlib import Path
import re
import zipfile

PAPER = Path(__file__).resolve().parents[1] / "paper"


def dependencies():
    files = {"acl.sty", "acl_natbib.bst", "numbers.tex", "Makefile",
             "check_arr.py", "page_limit.txt"}
    seen = set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        path = (PAPER / name).resolve()
        if not path.is_relative_to(PAPER.resolve()) or not path.is_file():
            raise FileNotFoundError(f"Missing manuscript dependency: {name}")
        files.add(name)
        text = re.sub(r"(?m)(?<!\\)%.*$", "", path.read_text())
        for kind, child in re.findall(r"\\(input|tabinput)\{([^{}]+)\}", text):
            if "#" in child:
                continue
            if kind == "tabinput":
                child = "tables/" + child
            if not Path(child).suffix:
                child += ".tex"
            visit(child)
        for asset in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", text):
            files.add(asset)
        for bibs in re.findall(r"\\bibliography\{([^{}]+)\}", text):
            files.update(bib + ".bib" for bib in bibs.split(","))

    visit("main.tex")
    for name in files:
        path = (PAPER / name).resolve()
        if not path.is_relative_to(PAPER.resolve()) or not path.is_file():
            raise FileNotFoundError(f"Missing manuscript dependency: {name}")
    return sorted(files)


def main():
    files = dependencies()
    output = PAPER / "submission_source.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in files:
            bundle.write(PAPER / name, name)
    print(f"Wrote {output}: {len(files)} required files")


if __name__ == "__main__":
    main()
