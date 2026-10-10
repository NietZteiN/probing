#!/usr/bin/env python
"""Prepare both review-file handoffs from the checked PDFs and minimal source archives."""
from pathlib import Path
import hashlib
import re
import shutil

WORKSPACE = Path(__file__).resolve().parents[2]


def main():
    for name, other in (('probing','codecue'),('codecue','probing')):
        root = WORKSPACE/name
        paper = root/'paper'
        destination = paper/'submission_ready'
        destination.mkdir(exist_ok=True)
        source = (paper/'main.tex').read_text()
        numbers = dict(re.findall(r'\\csname NUMval@([^\\]+)\\endcsname\{([^}]*)\}',
                                  (paper/'numbers.tex').read_text()))
        title = re.search(r'\\title\{([^}]+)\}',source).group(1)
        abstract = source.split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0]
        abstract = re.sub(r'\\NUM\{([^}]+)\}',lambda m:numbers[m.group(1)],abstract)
        abstract = abstract.replace(r'\citeauthor{kudo2026faithful}','Kudo et al.')
        abstract = ' '.join(abstract.replace(r'\%','%').replace('--','–').split())
        assert '\\' not in abstract and '{' not in abstract and '}' not in abstract
        if name=='codecue':
            assert ':' not in abstract and '—' not in abstract
        tldr = ('Writing arithmetic calculations makes answers less sensitive to misleading variable names.'
                if name=='probing' else
                'Models can write values suggested by misleading identifiers while correct digits remain decodable; source-expression examples reduce those errors.')
        area = 'Interpretability and Analysis of Models for NLP' if name=='probing' else 'NLP and Code Models'
        shutil.copyfile(paper/'main.pdf',destination/'paper.pdf')
        shutil.copyfile(paper/'submission_source.zip',destination/'source.zip')
        shutil.copyfile(WORKSPACE/other/'paper/main.pdf',destination/'concurrent_paper.pdf')
        shutil.copyfile(root/'docs/ARR_CHECKLIST.md',destination/'responsible_nlp_checklist.md')
        shutil.copyfile(root/'docs/SUBMISSION_READY.md',destination/'requirements_checklist.md')
        (destination/'abstract.txt').write_text(abstract+'\n')
        (destination/'submission_fields.md').write_text(
            f'# Submission fields\n\nTitle: {title}\n\nType: Short paper\n\nSuggested area: {area}\n\n'
            f'TL;DR: {tldr}\n\nAbstract:\n\n{abstract}\n\n'
            'Authors, submission history, preferred venue, preprint declaration and service contributor '
            'require author completion in OpenReview.\n')
        (destination/'README.md').write_text(
            '# Review submission files\n\n'
            'Upload `paper.pdf` as the paper. It includes references and necessary appendices.\n\n'
            'Upload `concurrent_paper.pdf` in the dedicated concurrent-submissions field if both '
            'papers are submitted in this cycle. The papers cite each other anonymously and '
            'describe their distinct contributions.\n\n'
            '`source.zip` is the clean LaTeX handoff, checked by standalone compilation. '
            '`submission_fields.md` and `abstract.txt` contain plain-text metadata. '
            '`responsible_nlp_checklist.md` contains evidence-backed draft form answers. '
            '`requirements_checklist.md` separates verified file requirements from author/account tasks.\n\n'
            'The arithmetic paper tests how written calculations affect final-answer sensitivity '
            'to number-word names. The code paper tests correct-value readouts before actual wrong '
            'trace writes and how source-expression examples change those writes.\n\n'
            'No OpenReview submission or author declaration has been made by this packaging script.\n')
        files = sorted(p for p in destination.iterdir() if p.name!='SHA256SUMS.txt')
        (destination/'SHA256SUMS.txt').write_text(''.join(
            f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
        print(name,'prepared',destination)


if __name__=='__main__':
    main()
