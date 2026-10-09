#!/usr/bin/env python
"""Validate and summarize the frozen response-control experiments after all workers finish."""
from __future__ import annotations

import itertools
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cueconf.config import OUT_DIR
from cueconf.generator import Instance
from cueconf.round6 import computation_key
from cueconf.round7 import FORMATS, parse_factorial, parse_final, digest, verified_calculation
from cueconf.stats import bootstrap_ci

BASE = OUT_DIR/'round7_controls'


def interval(rows, field, confidence=.95):
    if not rows:
        return None
    return list(bootstrap_ci(np.array([r[field] for r in rows], dtype=float),
                            np.array([r['program_key'] for r in rows]), n_boot=4000,
                            confidence=confidence))


def read_output(path, parser, experiment=None, budget=None):
    data = json.loads(path.read_text())
    if not data['complete']:
        raise ValueError('incomplete output')
    if experiment is not None and (data['experiment']!=experiment or
                                   data['max_new_tokens']!=budget or data['greedy'] is not True):
        raise ValueError('decoding differs from the frozen protocol')
    for row in data['records']:
        pred = parser(row['generation'])
        if pred != row['prediction'] or (pred == row['answer']) != row['correct'] or (pred is not None) != row['parsed']:
            raise ValueError('saved output disagrees with independent parsing')
    return data['records']


def paired_observations(records, role):
    def indexed(condition):
        selected=[r for r in records if r['condition']==condition and
                  (condition=='neutral' or r['target']==role)]
        index={(r['seed'],r['set_id']):r for r in selected}
        if len(index)!=len(selected):
            raise ValueError('duplicate paired observations')
        return index
    neutral, misleading=indexed('neutral'), indexed('incongruent')
    if not neutral or set(neutral)!=set(misleading):
        raise ValueError('ordinary and misleading cohorts differ')
    rows = []
    for key,r in misleading.items():
        n = neutral[key]
        if n['program_key']!=r['program_key'] or n['answer']!=r['answer']:
            raise ValueError('unmatched computations')
        rows.append({'seed':r['seed'], 'set_id':r['set_id'], 'program_key':r['program_key'],
            'excess':int(r['prediction']==r['lure'])-int(n['prediction']==r['lure']),
            'neutral_suggested':int(n['prediction']==r['lure']),
            'misleading_suggested':int(r['prediction']==r['lure']),
            'neutral_correct':int(n['correct']), 'misleading_correct':int(r['correct']),
            'neutral_parsed':int(n['parsed']), 'misleading_parsed':int(r['parsed'])})
    return rows


def summary(rows):
    return {'n_pairs':len(rows), 'n_computations':len({r['program_key'] for r in rows}),
            'pairs_by_seed':{str(seed):sum(r['seed']==seed for r in rows)
                             for seed in sorted({r['seed'] for r in rows})},
            'counts':{field:sum(r[field] for r in rows) for field in
                      ('neutral_suggested','misleading_suggested','neutral_correct',
                       'misleading_correct','neutral_parsed','misleading_parsed')},
            **{field+'_ci95':interval(rows,field) for field in
               ('excess','neutral_correct','misleading_correct','neutral_parsed','misleading_parsed')}}


def contrast(a, b):
    index={(r['seed'],r['set_id']):r for r in b}
    keys={(r['seed'],r['set_id']) for r in a}
    if not a or len(index)!=len(b) or len(keys)!=len(a) or keys!=set(index):
        raise ValueError('format comparison changed eligible pairs')
    if any(r['program_key']!=index[r['seed'],r['set_id']]['program_key'] for r in a):
        raise ValueError('format comparison changed arithmetic clusters')
    changes=[{**r, 'change':r['excess']-index[r['seed'],r['set_id']]['excess'],
              'accuracy_change':r['neutral_correct']-index[r['seed'],r['set_id']]['neutral_correct']}
             for r in a]
    accuracy90=interval(changes,'accuracy_change',.90)
    matched=(np.mean([r['neutral_correct'] for r in a])>=.9 and
             np.mean([r['neutral_correct'] for r in b])>=.9 and
             -.02<accuracy90[1] and accuracy90[2]<.02)
    return {'change_ci95':interval(changes,'change'), 'neutral_accuracy_change_ci90':accuracy90,
            'competence_matched':bool(matched),
            'change_by_seed':{str(seed):float(np.mean([r['change'] for r in changes if r['seed']==seed]))
                              for seed in sorted({r['seed'] for r in changes})}}


def main():
    cohort=json.loads((BASE/'cohorts.json').read_text())
    frozen=json.loads((ROOT/'log/round7_2026-10-09/frozen_manifest.json').read_text())
    if hashlib.sha256((BASE/'cohorts.json').read_bytes()).hexdigest()!=frozen['cohort_sha256']:
        raise ValueError('frozen factorial cohort changed')
    source_files={}
    def read(path, parser, experiment, budget):
        rows=read_output(path,parser,experiment,budget)
        source_files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return rows
    results={'validated':True,'protocol_sha256':cohort['protocol_sha256'],
             'cluster':'canonical arithmetic computation across names and demonstration repeats',
             'n_boot':4000,'interpretation':'Exploratory controls; all parse failures remain in denominators.',
             'interval_scope':'Pointwise exploratory intervals, without family-wide multiplicity correction. '
                              'A zero-event bootstrap interval is not a population bound.',
             'source_files_sha256':source_files,'models':{}}
    for model in cohort['models']:
        root=BASE/model
        if not json.loads((root/'complete.json').read_text())['complete']:
            raise ValueError(f'incomplete worker: {model}')
        selection=json.loads((root/'selection.json').read_text())
        eligible=json.loads((root/'final_cohort.json').read_text())
        if hashlib.sha256((root/'final_cohort.json').read_bytes()).hexdigest()!=frozen['models'][model]['frozen_eligibility_sha256']:
            raise ValueError('frozen generated-calculation eligibility changed')
        for pairs in eligible['groups'].values():
            for pair in pairs:
                for item in pair['pair']:
                    x=Instance(**item['instance'])
                    if computation_key(x)!=pair['program_key'] or verified_calculation(x,item['calculation']) is None:
                        raise ValueError('fixed calculation does not match executable gold')
        calibration_passes=[]
        expected_counts=[]
        for count,diagnostic in selection['calibration'].items():
            expected_counts.append(int(count))
            accuracy={}
            for style in FORMATS:
                saved=read(root/'calibration'/f'k{count}_{style}.json',parse_factorial,'formats',128)
                if [r['id'] for r in saved]!=[x['id'] for x in cohort['calibration']]:
                    raise ValueError('calibration cohort changed')
                if any(r['seed']!=7 or r['format']!=style or
                       any(r[field]!=x[field] for field in ('set_id','condition','target','lure','answer')) or
                       r['program_key']!=computation_key(Instance(**x))
                       for r,x in zip(saved,cohort['calibration'])):
                    raise ValueError('calibration executable labels changed')
                accuracy[style]=sum(r['correct'] for r in saved)/len(saved)
            if accuracy!=diagnostic['accuracy']:
                raise ValueError('calibration scores disagree with saved outputs')
            passed=min(accuracy.values())>=.90 and max(accuracy.values())-min(accuracy.values())<=.0200001
            if passed!=diagnostic['pass']:
                raise ValueError('calibration decision changed')
            if passed:calibration_passes.append(int(count))
        expected_stop=cohort['example_counts'].index(min(calibration_passes))+1 if calibration_passes else len(cohort['example_counts'])
        if expected_counts!=cohort['example_counts'][:expected_stop] or len(calibration_passes)>1:
            raise ValueError('calibration did not stop at the first passing count')
        if selection['count']!=(min(calibration_passes) if calibration_passes else 16):
            raise ValueError('example count was not selected by the frozen neutral rule')
        if selection['calibration_pass']!=bool(calibration_passes) or selection['selection_uses_test_outcomes'] is not False:
            raise ValueError('calibration selection flags changed')
        outputs={}
        for style in FORMATS:
            rows=[]
            for seed in cohort['seeds']:
                saved=read(root/'formats'/f's{seed}_{style}.json',parse_factorial,'formats',128)
                if [r['id'] for r in saved]!=[x['id'] for x in cohort['primary']]:
                    raise ValueError('factorial cohort changed')
                if any(r['seed']!=seed or r['format']!=style for r in saved):
                    raise ValueError('factorial labels changed')
                if any(any(r[field]!=x[field] for field in ('condition','target','lure','answer','set_id'))
                       for r,x in zip(saved,cohort['primary'])):
                    raise ValueError('factorial executable labels changed')
                if any(r['program_key']!=computation_key(Instance(**x)) for r,x in zip(saved,cohort['primary'])):
                    raise ValueError('factorial arithmetic clusters changed')
                rows+=saved
            outputs[style]=rows
        for seed in cohort['seeds']:
            lengths={}
            for style,rows in outputs.items():
                for r in rows:
                    if r['seed']==seed:
                        lengths.setdefault(r['set_id'],set()).add(r['prompt_tokens'])
            if any(len(values)!=1 for values in lengths.values()):
                raise ValueError('token-count matching failed')
        formats={}
        for role in ('v1','v2'):
            pairs={style:paired_observations(rows,role) for style,rows in outputs.items()}
            comparisons={}
            for a,b in (('no_names_equations','names_equations'),('no_names_values','names_values'),
                        ('names_equations','names_values'),('no_names_equations','no_names_values')):
                comparisons[a+' minus '+b]=contrast(pairs[a],pairs[b])
            formats[role]={'cells':{style:summary(rows) for style,rows in pairs.items()},
                           'comparisons':comparisons}
        final={}
        for role in ('v1','v2'):
            pairs={}
            for repeated,named in itertools.product((False,True),repeat=2):
                rows=[]
                for seed in cohort['seeds']:
                    path=root/'final_query'/f's{seed}_{role}_repeat{int(repeated)}_named{int(named)}.json'
                    saved=read(path,parse_final,'final_query',16)
                    expected=[item for pair in eligible['groups'][f's{seed}/{role}'] for item in pair['pair']]
                    if [r['id'] for r in saved]!=[item['instance']['id'] for item in expected]:
                        raise ValueError('fixed-calculation cohort changed')
                    if any(r['calculation_sha256']!=digest(item['calculation']) for r,item in zip(saved,expected)):
                        raise ValueError('fixed generated calculation changed')
                    if any(r['set_id']!=item['instance']['set_id'] or
                           r['program_key']!=computation_key(Instance(**item['instance']))
                           for r,item in zip(saved,expected)):
                        raise ValueError('fixed-calculation pairing or arithmetic clusters changed')
                    if any(r['seed']!=seed or r['target']!=role or r['repeat_name']!=repeated or r['named_output']!=named
                           or r['answer']!=item['instance']['answer'] or r['condition']!=item['instance']['condition']
                           for r,item in zip(saved,expected)):
                        raise ValueError('final-query executable labels or interventions changed')
                    if any(saved[2*i]['lure']!=pair['pair'][1]['instance']['lure'] or
                           saved[2*i+1]['lure']!=pair['pair'][1]['instance']['lure']
                           for i,pair in enumerate(eligible['groups'][f's{seed}/{role}'])):
                        raise ValueError('final-query suggested digit changed')
                    rows+=saved
                pairs[f'repeat{int(repeated)}_named{int(named)}']=paired_observations(rows,role)
            comparisons={}
            for named in (0,1):
                a,b=f'repeat1_named{named}',f'repeat0_named{named}'
                comparisons[a+' minus '+b]=contrast(pairs[a],pairs[b])
            for repeated in (0,1):
                a,b=f'repeat{repeated}_named1',f'repeat{repeated}_named0'
                comparisons[a+' minus '+b]=contrast(pairs[a],pairs[b])
            final[role]={'cells':{style:summary(rows) for style,rows in pairs.items()},
                         'comparisons':comparisons}
        results['models'][model]={'selection':selection,'formats':formats,'final_query':final,
            'runtime':json.loads((root/'complete.json').read_text()),
            'generated_lengths':{style:float(np.mean([r['generated_tokens'] for r in rows])) for style,rows in outputs.items()}}
    destination=ROOT/'results/summary/round7_response_controls.json'
    destination.write_text(json.dumps(results,indent=2)+'\n')
    print('Validated all format and final-query controls:',destination)
    runpy.run_path(str(ROOT/'scripts/89_response_control_tables.py'),run_name='__main__')


if __name__=='__main__':
    main()
