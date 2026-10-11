#!/usr/bin/env python
"""Refresh the combined manuscript from the two verified releases (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PAPER = ROOT / 'paper_combined'
SOURCES = {'a': ROOT / 'paper', 'c': WORKSPACE / 'codecue/paper'}
CODE_MODELS = [('olmo2-7b-it', 'OLMo-2-7B-I', 'olmo7'),
               ('llama32-3b-it', 'Llama-3.2-3B-I', 'llama3'),
               ('llama31-8b-it', 'Llama-3.1-8B-I', 'llama8')]
ARITH_MODELS = [('llama32-3b', 'Llama-3.2-3B'), ('llama32-3b-it', 'Llama-3.2-3B-I'),
                ('llama31-8b', 'Llama-3.1-8B'), ('llama31-8b-it', 'Llama-3.1-8B-I'),
                ('gemma3-4b-it', 'Gemma-3-4B-I'), ('gemma3-12b-it', 'Gemma-3-12B-I'),
                ('olmo2-1b-it', 'OLMo-2-1B-I*'), ('olmo2-7b-it', 'OLMo-2-7B-I')]


def load(repo, name):
    path = WORKSPACE / repo / 'results/summary' / f'{name}.json'
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_numbers():
    out = {}
    for prefix, source in SOURCES.items():
        for line in (source/'numbers.tex').read_text().splitlines():
            match = re.search(r'\\csname NUMval@([^\\]+)\\endcsname\{(.*)\}$', line)
            if match:
                out[prefix+'-'+match[1]] = match[2]
    comparison = load('codecue', 'round7_prompt_comparison')
    assert comparison['validated']
    for key, _, tag in CODE_MODELS:
        row = comparison['models'][key]['pooled']['lure_write']
        out[f'j-{tag}-error-writes'] = str(row['n'])
        out[f'j-{tag}-error-computations'] = str(row['n_programs'])
        for stage, field in [('prompt', 'prompt_accuracy_ci95'),
                             ('before-write', 'prewrite_accuracy_ci95'),
                             ('change', 'prewrite_minus_prompt_ci95')]:
            for suffix, value in zip(('mean', 'lo', 'hi'), row[field]):
                out[f'j-{tag}-{stage}-{suffix}'] = f'{100*value:.1f}'
    controls = load('probing', 'round7_response_controls')
    assert controls['validated']
    cells = controls['models']['olmo2-1b-it']['final_query']['v1']['cells']
    for style, tag in [('repeat0_named0', 'unnamed'), ('repeat1_named0', 'named')]:
        out[f'j-final-{tag}-accuracy'] = f"{100*cells[style]['neutral_correct_ci95'][0]:.1f}"
    correctness = json.loads((PAPER/'format_correctness.json').read_text())
    assert correctness['validated']
    for key, _, tag in CODE_MODELS:
        for regime, style in [('trace', 'values'), ('trace_expr', 'expression'),
                              ('expression_minus_values', 'change')]:
            row = correctness['models'][key][regime]
            for field, metric in [('neutral_step', 'ordinary-first'), ('misleading_step', 'misleading-first'),
                                   ('neutral_final', 'ordinary-final'), ('misleading_final', 'misleading-final')]:
                for suffix, value in zip(('mean', 'lo', 'hi'), row[field+'_correct_ci95']):
                    out[f'j-format-{tag}-{style}-{metric}-{suffix}'] = f'{100*value:.1f}'
    return out


def refresh_imports():
    seen = set()

    def transform(text, prefix):
        text = re.sub(r'\\NUM\{([^}]+)\}', lambda m: r'\NUM{'+prefix+'-'+m[1]+'}', text)
        text = re.sub(r'\\(label|ref|pageref)\{([^}]+)\}',
                      lambda m: '\\'+m[1]+'{'+prefix+'-'+m[2]+'}', text)
        text = re.sub(r'\\newsavebox\{[^}]+\}\s*', '', text)

        def child(match):
            kind, name = match.groups()
            path = ('tables/'+name if kind=='tabinput' else name)
            if not Path(path).suffix:
                path += '.tex'
            dest = import_file(prefix, path)
            return r'\input{'+dest.removesuffix('.tex')+'}'

        text = re.sub(r'\\(input|tabinput)\{([^{}]+)\}', child, text)
        text = re.sub(r'\\IfFileExists\{(tables/[^{}]+)\}',
                      lambda m: r'\IfFileExists{'+import_file(prefix,m[1])+'}',text)

        def asset(match):
            options, name = match.groups()
            destination = 'figures/'+prefix+'_'+Path(name).name
            shutil.copyfile(SOURCES[prefix]/name, PAPER/destination)
            return r'\includegraphics'+(options or '')+'{'+destination+'}'

        return re.sub(r'\\includegraphics(\[[^\]]*\])?\{([^{}]+)\}', asset, text)

    def import_file(prefix, path):
        dest = 'tables/'+prefix+'_'+Path(path).name
        if (prefix, path) not in seen:
            seen.add((prefix, path))
            (PAPER/dest).write_text(transform((SOURCES[prefix]/path).read_text(), prefix))
        return dest

    arithmetic = (SOURCES['a']/'appendix.tex').read_text()
    arithmetic = arithmetic.replace('\\section{Task and reproducibility}',
                                    '\\section{Arithmetic task and reproducibility}')
    arithmetic = transform(arithmetic, 'a')
    behavior = import_file('a', 'tables/behavior.tex')
    table = (r'\begin{table*}[t]'+'\n'+r'\centering\small'+'\n'+
             r'\resizebox{\textwidth}{!}{\input{'+behavior.removesuffix('.tex')+'}}\n'+
             r'\caption{Arithmetic level 3. Accuracy (\%); name effects (points). *Reliable across demonstration sets.}'+ '\n'+
             r'\label{a-tab:behavior}'+'\n'+r'\end{table*}'+'\n')
    arithmetic = arithmetic.replace(r'\section{Full behavioral results and controls}',
                                     r'\section{Full arithmetic results and controls}'+'\n'+table)
    # The original main-text patch curve remains a supporting figure here.
    shutil.copyfile(SOURCES['a']/'figures/story_patching.pdf', PAPER/'figures/a_story_patching.pdf')
    arithmetic += ('\n'+r'\begin{figure}[t]'+'\n'+r'\centering'+'\n'+
                   r'\includegraphics[width=\columnwidth]{figures/a_story_patching.pdf}'+'\n'+
                   r'\caption{Arithmetic name-site patches, Llama-3.2-3B, queried variable, set 7. All replaces every layer.}'+'\n'+
                   r'\label{a-fig:patch}'+'\n'+r'\end{figure}'+'\n')
    arithmetic = arithmetic.replace('across this study and its code companion', 'across the arithmetic and code experiments')
    arithmetic = arithmetic.replace('excluded from the main behavioral table',
                                     'excluded from the accuracy-filtered behavioral table')
    arithmetic += ('\n'+r'\begin{figure*}[t]'+ '\n'+r'\centering'+ '\n'+
                   r'\includegraphics[width=\textwidth]{figures/shared_behavior.pdf}'+ '\n'+
                   r'\caption{Selected name-error contrasts (95\% intervals); scales differ.}'+ '\n'+
                   r'\label{fig:behavior}\label{a-fig:master}'+ '\n'+r'\end{figure*}'+ '\n')
    (PAPER/'appendix/arithmetic.tex').write_text(arithmetic)

    source = (SOURCES['c']/'main.tex').read_text()
    code = source.split(r'\appendix',1)[1].split(r'\end{document}',1)[0]
    code = transform(code, 'c')
    code = code.replace(r'\section{Task levels and identifiers}', r'\section{Code task levels and identifiers}')
    central = import_file('c', 'tables/central_evidence.tex')
    code += ('\n'+r'\begin{table*}[t]'+'\n'+r'\centering\small'+'\n'+
             r'\input{'+central.removesuffix('.tex')+'}\n'+
             r'\caption{Code behavior and error-conditioned readouts [95\% interval]. Counts distinguish writes from computations.}'+ '\n'+
             r'\label{c-tab:central}'+'\n'+r'\end{table*}'+'\n')
    # The four baseline formats form a subset of the main six-format comparison.
    code = code.replace(r'Table~\ref{c-tab:formats} with 95\% bootstrap intervals',
                        r'Baseline formats with 95\% bootstrap intervals')
    formats = ('\n'+r'\begin{table*}[t]'+'\n'+r'\centering\small'+'\n'+
               r'\input{tables/shared_formats}'+'\n'+
               r'\caption{Code format comparison: added sum writes (points). Dashes: untested.}'+'\n'+
               r'\label{tab:formats}\label{c-tab:formats}'+'\n'+r'\end{table*}'+'\n')
    code = code.replace(r'\section{Demonstration formats}',
                        r'\section{Demonstration formats}'+formats)
    correctness = ('\n'+r'\section{Correctness on the matched code format cohort}'+'\n'+
                   r'\label{app:format-correctness}'+'\n'+
                   'We score the first written value and final answer on the same 285 original length-to-sum programs and their ordinary-name twins, under values-only and expression-and-value examples. Each format has 855 paired observations over three demonstration sets. Parse failures count as incorrect. Intervals use 4,000 bootstrap draws over matched program IDs, keeping all demonstration repeats and both names together. The format changes are paired on those same IDs. These are overall correctness rates within the specified cohort, rather than the larger task pool. Zero-width intervals record this sample, not certainty about population rates.\n'+
                   r'\begin{table*}[t]\centering\small'+'\n'+r'\input{tables/shared_correctness}'+'\n'+
                   r'\caption{Matched-code correctness and paired format changes (95\% intervals).}'+'\n'+
                   r'\label{tab:format-correctness}'+'\n'+r'\end{table*}'+'\n')
    code += correctness
    for name in ['code.tex']:
        (PAPER/'appendix'/name).write_text(code)
    for path in (PAPER/'tables').glob('c_*.tex'):
        text = path.read_text().replace('for this study and its arithmetic companion',
                                      'across the arithmetic and code experiments')
        path.write_text(text)


def formats_table(numbers):
    rows = [('Values only', r'\texttt{v = 2}', 'excess', None),
            ('Matched descriptive text', r'\texttt{v = 2 [notes]}', None, 'annotation'),
            ('Numeric elaboration', r'\texttt{v = 2 + 0 = 2}', None, 'numeric'),
            ('Interactive Python', r'\texttt{>>> v}\quad\texttt{2}', 'repl-excess', None),
            ('Code comments', r'\texttt{v = len(xs) \# 2}', 'comment-excess', None),
            ('Expression and value', r'\texttt{v = len(xs) = 2}', 'traceexpr-excess', None)]
    lines = [r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
             r'Example format & Shown value & OLMo-2-7B-I & Llama-3.2-3B-I & Llama-3.1-8B-I \\',r'\midrule']
    for label, example, cell, control in rows:
        values = []
        for _, _, tag in CODE_MODELS:
            if cell:
                key = f'c-cell-{tag}-{cell}'
                values.append(numbers[key])
            elif tag == 'llama8':
                values.append('--')
            else:
                values.append('+'+numbers[f'c-format-{tag}-{control}-mean'])
        lines.append(' & '.join([label,example,*values])+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (PAPER/'tables/shared_formats.tex').write_text('\n'.join(lines)+'\n')


def story_tables():
    num = lambda key: r'\NUM{'+key+'}'
    lines = [r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
             r'Model & \shortstack{Added sum writes\\Values only (points)} & \shortstack{Correct-digit readout (\%)\\Values-only errors\\Before wrong value} & \shortstack{Added sum writes\\Expressions (points)} \\',
             r'\midrule']
    for _, label, tag in CODE_MODELS:
        lines.append(' & '.join([label, num(f'c-cell-{tag}-excess'), num(f'j-{tag}-before-write-mean'),
                                num(f'c-cell-{tag}-traceexpr-excess')])+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (PAPER/'tables/shared_story.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{tabular}{@{}lrr@{}}',r'\toprule',
             r'Model & \shortstack{Correct first value (\%)\\Values $\to$ expressions} & \shortstack{Correct final answer (\%)\\Values $\to$ expressions} \\',r'\midrule']
    for _, label, tag in CODE_MODELS:
        rates = [num(f'j-format-{tag}-values-misleading-{metric}-mean')+r' $\to$ '+
                 num(f'j-format-{tag}-expression-misleading-{metric}-mean') for metric in ['first','final']]
        lines.append(' & '.join([label,*rates])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/shared_accuracy.tex').write_text('\n'.join(lines)+'\n')
    interval = lambda prefix: num(prefix+'-mean')+' ['+num(prefix+'-lo')+', '+num(prefix+'-hi')+']'
    lines = [r'\begin{tabular}{@{}lllrr@{}}', r'\toprule',
             r'Model & Examples & Names & Correct first value (\%) & Correct final answer (\%) \\', r'\midrule']
    for _, label, tag in CODE_MODELS:
        for style, example in [('values','Values only'),('expression','Expression + value')]:
            for condition, names in [('ordinary','Ordinary'),('misleading','Misleading')]:
                lines.append(' & '.join([label,example,names,*[interval(f'j-format-{tag}-{style}-{condition}-{metric}')
                                                              for metric in ['first','final']]])+r' \\')
        lines.append(r'\addlinespace')
    lines += [r'\midrule', r'\multicolumn{5}{l}{Expression minus values-only correctness (percentage points)} \\',
              r'\midrule']
    for _, label, tag in CODE_MODELS:
        for condition, names in [('ordinary','Ordinary'),('misleading','Misleading')]:
            lines.append(' & '.join([label,'Paired change',names,*[interval(f'j-format-{tag}-change-{condition}-{metric}')
                                                                  for metric in ['first','final']]])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/shared_correctness.tex').write_text('\n'.join(lines)+'\n')


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    red, blue, green, gray = '#B5321F', '#2B4A9E', '#157A55', '#65717C'
    provenance = {}

    def save(fig, name, payload):
        fig.savefig(PAPER/'figures'/f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate':None})
        fig.savefig(PAPER/'figures'/f'{name}.png', bbox_inches='tight', dpi=180)
        (PAPER/'figures'/f'{name}_data.json').write_text(json.dumps(payload,indent=2)+'\n')
        plt.close(fig)

    runpy.run_path(str(ROOT/'scripts/94_combined_example.py'))['main']()

    arithmetic = load('probing','seed_sweep_L3')
    code = load('codecue','cell_inference')['cells']
    fig = plt.figure(figsize=(7.2,3.5))
    left = fig.add_axes([.23,.19,.27,.60])
    right = fig.add_axes([.70,.19,.28,.60])
    fig.text(.02,.97,'(a) Arithmetic, queried variable',weight='bold',va='top')
    fig.text(.56,.97,'(b) Code, length named as sum',weight='bold',va='top')
    payload={'arithmetic':{},'code':{}}
    for y,(key,label) in enumerate(ARITH_MODELS):
        for regime,offset,color,marker,legend in [('direct',-.13,red,'o','Answer only'),('cot',.13,blue,'s','Calculations')]:
            row=arithmetic[f'{key}/{regime}']['contrasts']['lure_excess@v1']
            mean=100*row['pooled_mean']; lo,hi=[100*v for v in row['ci95']]
            left.errorbar(mean,y+offset,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt=marker,color=color,ms=4,capsize=2,label=legend if y==0 else None)
            payload['arithmetic'].setdefault(key,{})[regime]=row
    left.set_yticks(range(8),[label for _,label in ARITH_MODELS],fontsize=7.5)
    left.invert_yaxis(); left.set_xlim(-1,6); left.set_xticks([0,2,4,6])
    left.set_xlabel('Added digit answers (points)',fontsize=8)
    left.legend(loc='lower left',bbox_to_anchor=(-.45,1.04),frameon=False,ncol=2,fontsize=8)
    fig.text(.02,.02,'I: instruction-tuned. *Calculation accuracy below 90%.',fontsize=8)
    for y,(key,label,_) in enumerate(CODE_MODELS):
        for regime,offset,color,marker,legend in [('trace',-.12,red,'o','Values only'),('trace_expr',.12,blue,'s','Expression + value')]:
            row=code[f'L5/{key}/{regime}/len-to-sum']
            mean,lo,hi=[100*row[k] for k in ['excess','lo','hi']]
            right.errorbar(mean,y+offset,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt=marker,color=color,ms=4,capsize=2,label=legend if y==0 else None)
            payload['code'].setdefault(key,{})[regime]=row
    right.set_yticks(range(3),[label for _,label,_ in CODE_MODELS],fontsize=8)
    right.invert_yaxis(); right.set_ylim(2.65,-.65); right.set_xlim(-2,70);right.set_xticks([0,20,40,60])
    right.set_xlabel('Added sum writes (points)',fontsize=8)
    right.legend(loc='lower left',bbox_to_anchor=(-.40,1.04),frameon=False,ncol=1,fontsize=8)
    for ax in [left,right]:
        ax.axvline(0,color='.65',lw=.8);ax.grid(axis='x',alpha=.15);ax.tick_params(axis='y',length=0)
    save(fig,'shared_behavior',payload)

    data=load('codecue','round7_prompt_comparison')
    assert data['validated']
    fig,ax=plt.subplots(figsize=(7.2,2.05))
    for y,(key,label,_) in enumerate(CODE_MODELS):
        row=data['models'][key]['pooled']['lure_write']
        for field,offset,color,marker,label in [('prompt_accuracy_ci95',-.11,gray,'s','Before trace'),('prewrite_accuracy_ci95',.11,green,'o','Before wrong value')]:
            mean,lo,hi=[100*v for v in row[field]]
            ax.errorbar(mean,y+offset,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt=marker,color=color,ms=5,capsize=3,label=label if y==0 else None)
    labels = [f"{label}  ({data['models'][key]['pooled']['lure_write']['n_programs']} programs)"
              for key,label,_ in CODE_MODELS]
    ax.set_yticks(range(3),labels,fontsize=8.5);ax.invert_yaxis()
    ax.set_xlim(0,104);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',alpha=.15)
    ax.tick_params(axis='y',length=0);ax.set_xlabel('Correct digit recovered (%)')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout()
    save(fig,'shared_readouts',data['models'])
    runpy.run_path(str(ROOT/'scripts/92_figure_candidates.py'))['main']()
    for name in ['operation_controls','fresh_names']:
        for suffix in ['pdf','png']:
            shutil.copyfile(PAPER/'figure_candidates'/f'{name}.{suffix}',
                            PAPER/'figures'/f'{name}.{suffix}')
    for repo,names in [('probing',['seed_sweep_L3','round7_response_controls','round7_paired_prompting']),('codecue',['cell_inference','round5_formats','round6_crossfit_probes','round7_prompt_comparison','round6_identifier_replication'])]:
        for name in names:
            p=WORKSPACE/repo/'results/summary'/f'{name}.json'
            provenance[f'{repo}/{name}']=sha(p)
    provenance['combined/format_correctness'] = sha(PAPER/'format_correctness.json')
    (PAPER/'evidence_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def final_sources(numbers):
    text='\n'.join(p.read_text() for p in PAPER.rglob('*.tex') if p.name!='numbers.tex')
    keys=sorted(set(re.findall(r'\\NUM\{([^}]+)\}',text)))
    missing=set(keys)-numbers.keys()
    if missing:
        raise ValueError(f'Undefined quantities: {sorted(missing)}')
    lines=['% Generated from verified arithmetic and code releases; a-/c- preserve namespaces.']
    lines += [r'\expandafter\def\csname NUMval@'+key+r'\endcsname{'+numbers[key]+'}' for key in keys]
    lines += [r'\renewcommand{\NUM}[1]{\ifcsname NUMval@#1\endcsname\csname NUMval@#1\endcsname\else\textcolor{red}{$\langle\langle$\texttt{#1}$\rangle\rangle$}\fi}']
    (PAPER/'numbers.tex').write_text('\n'.join(lines)+'\n')
    cited=set()
    for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',text):
        cited.update(key.strip() for key in group.split(','))
    bib={}
    bib_sources = [source/'refs.bib' for source in SOURCES.values()] + [PAPER/'refs_extra.bib']
    for source in bib_sources:
        for entry in re.split(r'(?m)(?=^@)',source.read_text()):
            match=re.match(r'@\w+\{([^,]+),',entry)
            if match:
                bib.setdefault(match[1],entry.strip())
    if cited-set(bib):
        raise ValueError(f'Missing references: {cited-set(bib)}')
    assert not any(key.startswith('anon2026') for key in cited)
    (PAPER/'refs.bib').write_text('% Cited primary references for the combined manuscript.\n\n'+'\n\n'.join(bib[key] for key in sorted(cited))+'\n')
    labels=re.findall(r'\\label\{([^}]+)\}',text)
    refs=set(re.findall(r'\\(?:ref|pageref)\{([^}]+)\}',text))
    if refs-set(labels):
        raise ValueError(f'Missing cross-references: {sorted(refs-set(labels))}')
    if len(labels)!=len(set(labels)):
        raise ValueError('Duplicate combined labels')
    abstract=(PAPER/'main.tex').read_text().split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0]
    abstract=re.sub(r'\\NUM\{([^}]+)\}',lambda m:numbers[m[1]],abstract)
    abstract=abstract.replace(r'\citet{kudo2026faithful}','Kudo et al. (2026)')
    abstract=' '.join(abstract.replace(r'\%','%').replace('--','–').split())
    assert '\\' not in abstract and ':' not in abstract and '—' not in abstract
    (PAPER/'abstract.txt').write_text(abstract+'\n')
    print(f'Combined source: {len(keys)} verified quantities, {len(cited)} references')


def bundle():
    package=runpy.run_path(str(ROOT/'scripts/84_submission_bundle.py'))
    package['dependencies'].__globals__['PAPER']=PAPER
    files=package['dependencies']()
    with zipfile.ZipFile(PAPER/'submission_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(PAPER/name,name)
    print(f'Combined source bundle: {len(files)} files')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',action='store_true')
    args=parser.parse_args()
    if args.bundle:
        bundle();return
    for name in ['figures','tables','appendix']:
        (PAPER/name).mkdir(parents=True,exist_ok=True)
    for name in ['acl.sty','acl_natbib.bst','Makefile']:
        shutil.copyfile(SOURCES['a']/name,PAPER/name)
    checker=(SOURCES['a']/'check_arr.py').read_text()
    checker=checker.replace('source.find(r"\\section{Conclusion}")','source.find(r"\\section{Discussion and conclusion}")')
    (PAPER/'check_arr.py').write_text(checker)
    (PAPER/'page_limit.txt').write_text('8\n')
    runpy.run_path(str(ROOT/'scripts/93_combined_correctness.py'))['main']()
    numbers=source_numbers()
    refresh_imports()
    formats_table(numbers)
    story_tables()
    figures()
    runpy.run_path(str(ROOT/'scripts/95_combined_demonstrations.py'))['main']()
    final_sources(numbers)


if __name__=='__main__':
    main()
