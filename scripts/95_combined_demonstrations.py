#!/usr/bin/env python
"""Typeset the actual code demonstration set used by the opening example."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/'codecue/src'))
from codecue.prompts import demos, demo_block


def tex(text):
    text = text.replace('_', r'\_').replace('#', r'\#')
    text = text.replace(', ', r',\allowbreak{} ')
    return r'\texttt{' + text + '}'


def main():
    lines = [r'\begin{table*}[t]',r'\centering\small',
             r'\begin{tabular}{@{}p{.32\textwidth}p{.25\textwidth}p{.37\textwidth}@{}}',
             r'\toprule',r'Program and input & Values-only trace & Expression-and-value trace \\',r'\midrule']
    for d in demos(5,7,'trace'):
        program = r'\newline '.join((r'\hspace*{1em}' if ln.startswith('    ') else '')+tex(ln.strip())
                                    for ln in d.program.splitlines())
        outputs = []
        for style in ('trace','trace_expr'):
            output = demo_block(d,style).split('\nTrace:',1)[1].strip()
            outputs.append(r'\newline '.join(tex(ln) for ln in output.splitlines()))
        lines.append(' & '.join([program,*outputs])+r' \\')
        lines.append(r'\addlinespace')
    lines += [r'\bottomrule',r'\end{tabular}',
              r'\caption{Complete code demonstration set 7; prepend \texttt{Trace:} to each trace.}',
              r'\label{tab:code-demonstrations}',r'\end{table*}']
    (ROOT/'paper_combined/tables/shared_demonstrations.tex').write_text('\n'.join(lines)+'\n')
    print('Typeset three verified code demonstration blocks.')


if __name__ == '__main__':
    main()
