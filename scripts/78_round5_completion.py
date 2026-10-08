#!/usr/bin/env python
"""Validate follow-ups and rebuild; incomplete GPU runs never receive completed boxes."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT.parent/'codecue'
sys.path[:0]=[str(ROOT/'src'),str(CODE/'src')]
from cueconf.config import OUT_DIR, RESULTS_DIR
from cueconf.followups import fixed_sample
from cueconf.generator import read_jsonl
from codecue.stats import bootstrap_ci
COUT=Path('/scratch/juno/jvl210002/codecue')
LOG=ROOT/'log/round5_2026-10-03'

def read(p):return [json.loads(l) for l in p.open()]
def interval(rows,key):return list(bootstrap_ci([r[key] for r in rows],[r['set_id'] for r in rows],n_boot=4000)) if rows else None

def generated():
 sample=fixed_sample(x.set_id for x in read_jsonl(ROOT/'data/L3/test_sets.jsonl'))
 cells=[]
 for model in ('olmo2-1b-it','llama32-3b'):
  for seed in (7,11,13):
   regime='cot' if seed==7 else f'cot_s{seed}'
   for role in ('v1','v2'):
    source=OUT_DIR/'round5_generated'/model/regime/f'{role}.json';x=json.loads(source.read_text())
    if not x['complete'] or x['n_train']!=10000 or x['probe_seeds']!=[0,1,2] or set(x['sample_sets'])!=sample or not x['pre_value_boundary_checked']:
     raise ValueError(f'incomplete generated readout {source}')
    rows=x['records'];lookup={(r['set_id'],r['condition']):r for r in rows}
    if len(lookup)!=len(rows) or any((id_,condition) not in lookup for id_ in sample for condition in ('neutral','incongruent')):
     raise ValueError('missing primary generated sample rows')
    for row in rows:
     if row['boundary_valid']:
      if len(row['predictions'])!=3 or not all(0<=p<=9 for p in row['predictions']) or not all(math.isfinite(row[k]) for k in ('probe_accuracy','p_true','control_accuracy')):
       raise ValueError('invalid generated probe predictions')
      if row['condition']=='incongruent' and not math.isfinite(row['margin']):raise ValueError('nonfinite generated margin')
    neutral=[r for r in rows if r['primary_sample'] and r['condition']=='neutral']
    valid=[r for r in neutral if r['boundary_valid']]
    population=[r for r in rows if r['primary_sample'] and r['condition']=='incongruent']
    lure=[r for r in population if r['boundary_valid'] and r['written_value']==r['lure']]
    extra=[r for r in rows if not r['primary_sample'] and r['condition']=='incongruent' and r['lure_augmentation'] and r['boundary_valid']]
    calibration=interval(valid,'probe_accuracy')
    cells.append({'model':model,'demo_seed':seed,'role':role,'layer':x['layer'],
     'n_primary_neutral':len(neutral),'n_primary_neutral_valid':len(valid),
     'neutral_calibration_ci95':calibration,'calibration_passes':bool(calibration and calibration[0]>=.9),
     'n_primary_incongruent':len(population),'n_primary_incongruent_valid':sum(r['boundary_valid'] for r in population),
     'n_primary_lure_valid':len(lure),'primary_lure_probe_ci95':interval(lure,'probe_accuracy'),
     'n_augmented_lure_valid':len(extra),'augmented_lure_probe_ci95':interval(extra,'probe_accuracy'),
     'primary_sample_probe_ci95':interval([r for r in population if r['boundary_valid']],'probe_accuracy'),
     'source':str(source),'n_missing_boundary':sum(not r['boundary_found'] for r in rows),
     'n_excluded_boundary':sum(r['boundary_found'] and not r['boundary_valid'] for r in rows)})
 return {'validated':True,'cells':cells,'n_cells':12,'calibration_rule':'>=90% held-out neutral generated accuracy per cell; other cells descriptive',
  'sample_sets':sorted(sample),'augmentation_rule':'error-selected extra cases, excluded from population estimates',
  'inference':'95% matched-program bootstrap, 4000 draws; three probe seeds averaged'}

def formats():
 from transformers import AutoTokenizer
 from codecue.config import model_entry
 from codecue.followups import control_demo
 from codecue.generator import read_jsonl as code_instances
 from codecue.prompts import demos, build_prompt
 instances={x.id:x for x in code_instances(CODE/'data/L5/test_sets.jsonl')}
 result={'validated':True,'models':{},'demonstration_sets':[7,11,13],'n_pairs':285}
 for model in ('olmo2-7b-it','llama32-3b-it'):
  tok=AutoTokenizer.from_pretrained(model_entry(model)['hf_id'],local_files_only=True)
  result['models'][model]={}
  for kind in ('neutral_annotation','numeric_elaboration'):
   values=[];seeds={}
   for seed in (7,11,13):
    root=COUT/'round5_formats'/model/f's{seed}'/kind
    meta=json.loads((root/'summary.json').read_text());rows=read(root/'behavior.jsonl')
    if meta['n']!=570 or meta['n_pairs']!=285 or len(rows)!=570 or [r['id'] for r in rows]!=meta['ids'] or not meta['greedy'] or meta['max_new_tokens']!=160:
     raise ValueError('incomplete format output')
    if kind=='neutral_annotation' and any(r['source_expression']!=r['control'] for r in meta['demo_counts']):
     raise ValueError('annotation token mismatch')
    ds=demos(5,seed,'trace')
    prefix='\n\n'.join(control_demo(d,kind,tok) for d in ds)
    prompts=[prefix+'\n\n'+instances[r['id']].program+'\nTrace:' for r in rows]
    if hashlib.sha256(json.dumps(prompts).encode()).hexdigest()!=meta['fingerprint']:
     raise ValueError('saved control prompt fingerprint mismatch')
    if kind=='neutral_annotation':
     reference=[build_prompt(instances[r['id']],'trace_expr',ds) for r in rows]
     counts=[len(ids) for ids in tok(prompts,add_special_tokens=True)['input_ids']]
     expected=[len(ids) for ids in tok(reference,add_special_tokens=True)['input_ids']]
     if counts!=expected:raise ValueError('full evaluation prompt lengths do not match')
    inc=[r for r in rows if r['condition']=='incongruent'];neutral={r['set_id']:r for r in rows if r['condition']=='neutral'}
    if len(inc)!=285 or len(neutral)!=285 or len({r['set_id'] for r in inc})!=285:raise ValueError('missing matched control rows')
    baselines={}
    for format_ in ('trace','trace_expr'):
     regime=format_ if seed==7 else f'{format_}_s{seed}'
     base=COUT/'runs'/model/'L5'/regime
     baselines[format_]=({r['set_id']:r for r in read(base/'incongruent@v1/behavior.jsonl')},
                        {r['set_id']:r for r in read(base/'neutral/behavior.jsonl')})
    seedrows=[]
    for row in inc:
     twin=neutral[row['set_id']];lure=row['lure']
     delta=int(row['value_written']==lure)-int(twin['values_written']['v1']==lure)
     value={'set_id':row['set_id'],'seed':seed,'delta':delta}
     for format_,(i,n) in baselines.items():
      original=int(i[row['set_id']]['value_written']==lure)-int(n[row['set_id']]['values_written']['v1']==lure)
      value['difference_vs_'+format_]=delta-original
     values.append(value);seedrows.append(value)
    seeds[str(seed)]={k:interval(seedrows,k) for k in ('delta','difference_vs_trace','difference_vs_trace_expr')}
   result['models'][model][kind]={'excess_ci95':interval(values,'delta'),
    'difference_vs_trace_ci95':interval(values,'difference_vs_trace'),
    'difference_vs_trace_expr_ci95':interval(values,'difference_vs_trace_expr'),'by_seed':seeds,'n':len(values),
    'prompt_fingerprints_checked':True,'full_prompt_token_count_checks':1710 if kind=='neutral_annotation' else 0}
 return result

def stage(report,argv,cwd=ROOT):
 path=LOG/f'publish_stage{len(report["stages"])+1}.out'
 with path.open('w') as f:run=subprocess.run(argv,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 report['stages'].append({'argv':argv,'exit_code':run.returncode,'log':str(path)})
 if run.returncode:raise RuntimeError(f'{argv} failed; see {path}')

def tick(item):
 path=ROOT/'CHECKLIST.md';s=path.read_text();s=re.sub(rf'(?m)^- \[ \] (?={item}:)', '- [x] ',s);path.write_text(s)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
 report={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'needs_attention','analyses':{},'stages':[],'errors':[]}
 for name,fn,root,item in (('generated',generated,ROOT,'R5-A4'),('formats',formats,CODE,'R5-A5')):
  try:
   x=fn();(root/'results/summary'/f'round5_{name}.json').write_text(json.dumps(x,indent=2)+'\n');report['analyses'][item]=True
  except Exception as error:
   report['analyses'][item]=False;report['errors'].append({'experiment':item,'error':str(error)})
 for name,root,item in (('error_probes',CODE,'R5-A1'),('identifiers',CODE,'R5-A2'),('competence',ROOT,'R5-A3')):
  try:
   x=json.loads((root/'results/summary'/f'round5_{name}.json').read_text())
   if not x['validated'] or (name=='competence' and x['n_models']!=9):raise ValueError('expected all nine arithmetic models')
   report['analyses'][item]=True
  except Exception as error:
   report['analyses'][item]=False;report['errors'].append({'experiment':item,'error':str(error)})
 try:
  stage(report,[sys.executable,'scripts/51_numbers.py'],CODE)
  stage(report,[sys.executable,'scripts/51_tables.py'])
  stage(report,[sys.executable,'scripts/99_selfcheck.py'],CODE)
  stage(report,[sys.executable,'scripts/99_selfcheck.py'])
  stage(report,['make','paper-submission'],CODE)
  stage(report,['make','paper-submission'])
  for item,complete in report['analyses'].items():
   if complete:tick(item)
  if all(report['analyses'].values()):tick('R5-P');report['status']='complete'
  else:report['status']='partial'
 except Exception as error:report['errors'].append({'publication':str(error)})
 (RESULTS_DIR/'summary/round5_completion.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)
 return 0 if report['status']=='complete' or (args.allow_partial and report['status']=='partial') else 1
if __name__=='__main__':sys.exit(main())
