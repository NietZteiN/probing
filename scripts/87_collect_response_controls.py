#!/usr/bin/env python
"""Validate and summarize the frozen response-control experiments after all workers finish."""
from __future__ import annotations

import itertools
import hashlib
import json
from pathlib import Path
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


def read_output(path, parser):
    data = json.loads(path.read_text())
    if not data['complete']:
        raise ValueError('incomplete output')
    for row in data['records']:
        pred = parser(row['generation'])
        if pred != row['prediction'] or (pred == row['answer']) != row['correct'] or (pred is not None) != row['parsed']:
            raise ValueError('saved output disagrees with independent parsing')
    return data['records']


def paired_observations(records, role):
    neutral = {(r['seed'],r['set_id']):r for r in records if r['condition']=='neutral'}
    rows = []
    for r in records:
        if r['condition']!='incongruent' or r['target']!=role:
            continue
        n = neutral[r['seed'],r['set_id']]
        if n['program_key']!=r['program_key'] or n['answer']!=r['answer']:
            raise ValueError('unmatched computations')
        rows.append({'seed':r['seed'], 'set_id':r['set_id'], 'program_key':r['program_key'],
            'excess':int(r['prediction']==r['lure'])-int(n['prediction']==r['lure']),
            'neutral_correct':int(n['correct']), 'misleading_correct':int(r['correct']),
            'neutral_parsed':int(n['parsed']), 'misleading_parsed':int(r['parsed'])})
    return rows


def summary(rows):
    return {'n_pairs':len(rows), 'n_computations':len({r['program_key'] for r in rows}),
            **{field+'_ci95':interval(rows,field) for field in
               ('excess','neutral_correct','misleading_correct','neutral_parsed','misleading_parsed')}}


def contrast(a, b):
    index={(r['seed'],r['set_id']):r for r in b}
    if len(index)!=len(b) or {(r['seed'],r['set_id']) for r in a}!=set(index):
        raise ValueError('format comparison changed eligible pairs')
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
    results={'validated':True,'protocol_sha256':cohort['protocol_sha256'],
             'cluster':'canonical arithmetic computation across names and demonstration repeats',
             'n_boot':4000,'interpretation':'Exploratory controls; all parse failures remain in denominators.',
             'models':{}}
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
        for count,diagnostic in selection['calibration'].items():
            accuracy={}
            for style in FORMATS:
                saved=read_output(root/'calibration'/f'k{count}_{style}.json',parse_factorial)
                if [r['id'] for r in saved]!=[x['id'] for x in cohort['calibration']]:
                    raise ValueError('calibration cohort changed')
                accuracy[style]=sum(r['correct'] for r in saved)/len(saved)
            if accuracy!=diagnostic['accuracy']:
                raise ValueError('calibration scores disagree with saved outputs')
            passed=min(accuracy.values())>=.90 and max(accuracy.values())-min(accuracy.values())<=.0200001
            if passed!=diagnostic['pass']:
                raise ValueError('calibration decision changed')
            if passed:calibration_passes.append(int(count))
        if selection['count']!=(min(calibration_passes) if calibration_passes else 16):
            raise ValueError('example count was not selected by the frozen neutral rule')
        outputs={}
        for style in FORMATS:
            rows=[]
            for seed in cohort['seeds']:
                saved=read_output(root/'formats'/f's{seed}_{style}.json',parse_factorial)
                if [r['id'] for r in saved]!=[x['id'] for x in cohort['primary']]:
                    raise ValueError('factorial cohort changed')
                if any(r['seed']!=seed or r['format']!=style for r in saved):
                    raise ValueError('factorial labels changed')
                if any(any(r[field]!=x[field] for field in ('condition','target','lure','answer','set_id'))
                       for r,x in zip(saved,cohort['primary'])):
                    raise ValueError('factorial executable labels changed')
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
                    saved=read_output(path,parse_final)
                    expected=[item for pair in eligible['groups'][f's{seed}/{role}'] for item in pair['pair']]
                    if [r['id'] for r in saved]!=[item['instance']['id'] for item in expected]:
                        raise ValueError('fixed-calculation cohort changed')
                    if any(r['calculation_sha256']!=digest(item['calculation']) for r,item in zip(saved,expected)):
                        raise ValueError('fixed generated calculation changed')
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


if __name__=='__main__':
    main()
