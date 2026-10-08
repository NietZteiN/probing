#!/usr/bin/env python
"""Validate round-six outputs and collect them without changing current manuscript numbers."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT.parent/'codecue'
sys.path[:0]=[str(ROOT/'src'),str(CODE/'src')]
from cueconf.config import OUT_DIR, RESULTS_DIR
from cueconf.round6 import computation_key
from cueconf.generator import Instance
from codecue.config import OUT_DIR as COUT, RESULTS_DIR as CRESULTS
from codecue.prompts import build_prompt,demos,value_written,parse_answer
from codecue.round6 import select_identifiers
from codecue.stats import bootstrap_ci

LOG=ROOT/'log/round6_2026-10-03'


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def interval(rows,field):
    if not rows:return None
    values=[r[field] for r in rows]
    if not np.isfinite(values).all():raise ValueError('nonfinite collected values')
    return list(bootstrap_ci(values,[r['program_key'] for r in rows],n_boot=4000))


def screened_identifiers(manifest):
    cfg=manifest['identifier_replication'];base=COUT/'round6_identifiers'
    gates={m:json.loads((base/'screen'/f'{m}.json').read_text()) for m in cfg['panel']}
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    for model,gate in gates.items():
        if not gate['complete'] or gate['model']!=model or gate['protocol_signature']!=signature:
            raise ValueError('meaning-screen protocol mismatch')
        for context in ('canonical','calculate','process'):
            for name in cfg['candidates']+['v','w','zz']:
                scores=gate[context][name]['scores']
                if not np.isfinite(list(scores.values())).all() or gate[context][name]['argmax']!=max(scores,key=scores.get):
                    raise ValueError('invalid meaning-screen scores')
    names,scores=select_identifiers(gates,cfg['panel'],cfg['candidates'])
    frozen=json.loads((base/'frozen_names.json').read_text())
    if names!=frozen['selected_names'] or scores!=frozen['scores']:
        raise ValueError('frozen names disagree with original panel gate')
    if any(frozen['screen_sha256'][m]!=hashlib.sha256((base/'screen'/f'{m}.json').read_bytes()).hexdigest() for m in cfg['panel']):
        raise ValueError('meaning-screen files changed after name selection')
    result={'validated':True,'selected_names':names,'gate':scores,'models':{},'records':[],
            'status':'complete' if names else 'unavailable','interpretation':'Independent exploratory replication; original protocol deviation remains disclosed.'}
    if not names:return result
    spec=importlib.util.spec_from_file_location('replication',CODE/'scripts/79_screened_replication.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fresh=json.loads((base/'fresh_programs.json').read_text())['programs']
    if fresh!=module.fresh_programs():raise ValueError('fresh-program disjointness or fixed sample changed')
    instances=module.instances(names)
    for model in cfg['behavior_models']:
        pooled=[]
        for seed in cfg['demo_seeds']:
            for fmt in cfg['formats']:
                saved=json.loads((base/'runs'/model/f's{seed}'/f'{fmt}.json').read_text())
                prompts=[build_prompt(x,fmt,demos(5,seed,fmt)) for x in instances]
                fingerprint=hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
                if not saved['complete'] or saved['fingerprint']!=fingerprint or [r['id'] for r in saved['records']]!=[x.id for x in instances]:
                    raise ValueError('replication prompt or row mismatch')
                neutral={r['set_id']:r for r in saved['records'] if r['condition']=='neutral'}
                for x,row in zip(instances,saved['records']):
                    if row['written_value']!=value_written(row['generation'],x.names['v1']) or row['prediction']!=parse_answer(row['generation']):
                        raise ValueError('replication parsing changed')
                    if row['true_value']!=x.values['v1'] or row['lure']!=x.lure:
                        raise ValueError('replication executable labels changed')
                    if row['condition']=='neutral':continue
                    twin=neutral[row['set_id']]
                    pooled.append({'model':model,'seed':seed,'format':fmt,'name':row['name'],
                                   'set_id':row['set_id'],'program_key':row['program_key'],
                                   'delta':int(row['written_value']==row['lure'])-int(twin['written_value']==row['lure'])})
        summary={}
        for fmt in cfg['formats']:
            rows=[r for r in pooled if r['format']==fmt]
            summary[fmt]={'excess_ci95':interval(rows,'delta'),
                          'by_name':{name:interval([r for r in rows if r['name']==name],'delta') for name in names},
                          'by_seed':{str(seed):interval([r for r in rows if r['seed']==seed],'delta') for seed in cfg['demo_seeds']}}
        trace={(r['set_id'],r['name'],r['seed']):r for r in pooled if r['format']=='trace'}
        paired=[{**r,'change':r['delta']-trace[r['set_id'],r['name'],r['seed']]['delta']} for r in pooled if r['format']=='trace_expr']
        summary['expression_minus_trace_ci95']=interval(paired,'change')
        summary['n_programs']=200;summary['n_names']=len(names)
        result['models'][model]=summary;result['records']+=pooled
    return result


def generated_probes(manifest):
    base=OUT_DIR/'round6_generated';cohorts=json.loads((base/'cohorts.json').read_text())
    keys={split:{computation_key(Instance(**row)) for row in cohorts[split]} for split in ('train','validation')}
    if keys['train'] & keys['validation']:raise ValueError('training/validation computation overlap')
    result={'validated':True,'training':'neutral generated chains, demo7, true executed labels',
            'layer_selection':'independent neutral validation accuracy',
            'source_cohorts':{'train_n':len(cohorts['train']),'validation_n':len(cohorts['validation']),
                              'train_keys':len(keys['train']),'validation_keys':len(keys['validation'])},
            'models':{},'records':[]}
    for model in manifest['generated_probes']['models']:
        root=base/model
        if not json.loads((root/'complete.json').read_text())['complete']:raise ValueError('incomplete generated-chain probes')
        summary={}
        for role in ('v1','v2'):
            selection=json.loads((root/f'{role}_selection.json').read_text())
            scores=selection['validation_scores']
            if selection['layer']!=max(scores,key=lambda r:r['accuracy'])['layer']:
                raise ValueError('layer selection differs from independent validation rule')
            for split in ('train','validation'):
                meta=json.loads((root/split/role/'meta.json').read_text())
                if not meta['complete'] or not meta['pre_value_boundary_checked'] or meta['storage_dtype']!='float32':
                    raise ValueError('invalid generated training/validation cache')
                if any(r['program_key'] not in keys[split] for r in meta['records']):
                    raise ValueError('generated source cohort differs')
                if selection[f'{split}_signature']!=meta['signature']:
                    raise ValueError('generated probe weight source differs')
            cells={}
            for seed in (7,11,13):
                path=root/'test'/f's{seed}'/role
                test=json.loads((path/'meta.json').read_text());data=json.loads((path/'readouts.json').read_text())
                if not data['complete'] or data['test_signature']!=test['signature'] or data['weight_signature']!=selection['signature']:
                    raise ValueError('generated test cache/probe signature differs')
                if any(r['program_key'] in keys['train']|keys['validation'] for r in data['records']):
                    raise ValueError('generated training/validation computation overlaps test')
                regime='cot' if seed==7 else f'cot_s{seed}'
                old=json.loads((OUT_DIR/'round5_generated'/model/regime/f'{role}.json').read_text())
                old_rows={r['id']:r for r in old['records']}
                if [r['id'] for r in data['records']]!=[r['id'] for r in old['records']]:
                    raise ValueError('generated test sample changed')
                rows=[]
                for row in data['records']:
                    r={**row,'model':model,'role':role,'seed':seed}
                    if r['boundary_valid']:
                        if len(r['predictions'])!=3 or not np.isfinite([r['probe_accuracy'],r['control_accuracy']]).all():
                            raise ValueError('invalid generated probe predictions')
                        if not np.isclose(r['probe_accuracy'],np.mean(np.asarray(r['predictions'])==r['true_value'])):
                            raise ValueError('generated readout accuracy mismatch')
                        old_row=old_rows[r['id']]
                        if old_row['boundary_valid']:
                            r['gold_transfer_accuracy']=old_row['probe_accuracy']
                            r['change']=r['probe_accuracy']-old_row['probe_accuracy']
                    rows.append(r)
                valid=[r for r in rows if r['boundary_valid']]
                neutral=[r for r in valid if r['primary_sample'] and r['condition']=='neutral']
                calibration=interval(neutral,'probe_accuracy')
                strata={}
                for group in ('primary_sample','lure_augmentation'):
                    selected=[r for r in valid if r[group] and r['condition']=='incongruent' and r['written_value']==r['lure']]
                    if group=='lure_augmentation':selected=[r for r in selected if not r['primary_sample']]
                    strata[group]={'n':len(selected),'n_programs':len({r['program_key'] for r in selected}),
                                   'accuracy_ci95':interval(selected,'probe_accuracy'),
                                   'change_from_gold_ci95':interval([r for r in selected if 'change' in r],'change')}
                cells[str(seed)]={'n_total':len(rows),'n_valid':len(valid),'neutral_calibration_ci95':calibration,
                                  'calibration_pass':bool(calibration and calibration[0]>=.9),
                                  'neutral_control_ci95':interval(neutral,'control_accuracy'),
                                  'neutral_change_from_gold_ci95':interval([r for r in neutral if 'change' in r],'change'),
                                  'lure_writes':strata}
                result['records']+=rows
            summary[role]={'selection':selection,'cells':cells}
        result['models'][model]=summary
    return result


def main():
    manifest=json.loads((LOG/'manifest.json').read_text())
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'collecting','analyses':{},'errors':[]}
    for label,fn,target in [
        ('R6-A1',lambda:json.loads((CRESULTS/'summary/round6_crossfit_probes.json').read_text()),None),
        ('R6-A2',lambda:screened_identifiers(manifest),CRESULTS/'summary/round6_identifier_replication.json'),
        ('R6-A3',lambda:generated_probes(manifest),RESULTS_DIR/'summary/round6_generated.json')]:
        try:
            data=fn()
            if not data['validated']:raise ValueError('analysis did not validate')
            if target:write(target,data)
            report['analyses'][label]=True
            checklist=ROOT/'CHECKLIST.md';text=checklist.read_text()
            checklist.write_text(text.replace(f'- [ ] {label}:',f'- [x] {label}:'))
            print(label,'validated',flush=True)
        except Exception as error:
            report['analyses'][label]=False;report['errors'].append({'analysis':label,'error':str(error)})
            print(label,'FAILED',error,flush=True)
    report['status']='complete' if not report['errors'] else 'needs_attention'
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    write(RESULTS_DIR/'summary/round6_completion.json',report)
    manifest['status']=report['status'];manifest['collection']=report
    write(LOG/'manifest.json',manifest)
    if report['status']=='complete':
        path=ROOT/'CHECKLIST.md';path.write_text(path.read_text().replace('- [ ] R6-P:','- [x] R6-P:'))
    if report['errors']:sys.exit(1)


if __name__=='__main__':main()
