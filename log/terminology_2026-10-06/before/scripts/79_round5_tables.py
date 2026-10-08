#!/usr/bin/env python
"""Render validated follow-ups as appendix evidence, preserving descriptive limits."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
LABEL={'olmo2-7b-it':'OLMo-2-7B','llama32-3b-it':'Llama-3.2-3B-I','llama31-8b-it':'Llama-3.1-8B-I',
 'olmo2-1b-it':'OLMo-2-1B','gemma3-4b-it':'Gemma-3-4B-I','gemma3-4b':'Gemma-3-4B',
 'gemma3-12b-it':'Gemma-3-12B-I','llama32-3b':'Llama-3.2-3B','llama31-8b':'Llama-3.1-8B'}

def load(name):
 p=ROOT/'results/summary'/f'round5_{name}.json'
 if not p.exists():return None
 x=json.loads(p.read_text())
 if not x.get('validated'):raise ValueError(f'unvalidated follow-up {p}')
 return x

def ci(x):return '--' if x is None else f'{100*x[0]:.1f} [{100*x[1]:.1f}, {100*x[2]:.1f}]'
def number(x):return '--' if x is None else f'{x:+.2f}'
def narrative_numbers():
 out={}
 x=load('competence')
 if x:
  out['competence-models']=str(x['n_models'])
  out['competence-pairs']=str(x['n_pairs'])
  key='misleading_accuracy__lure_excess'
  out['competence-probe-lure-r']=f"{x['correlations'][key]['pearson']:+.3f}"
  out['competence-probe-lure-without-olmo-r']=f"{x['leave_one_model_out']['olmo2-1b-it'][key]['pearson']:+.3f}"
 x=load('generated')
 if x:
  for model,tag in (('llama32-3b','llama3'),('olmo2-1b-it','olmo1')):
   cells=[r for r in x['cells'] if r['model']==model]
   rates=[r['neutral_calibration_ci95'][0] for r in cells if r['neutral_calibration_ci95']]
   out[f'generated-{tag}-cal-lo']=f'{100*min(rates):.1f}'
   out[f'generated-{tag}-cal-hi']=f'{100*max(rates):.1f}'
   out[f'generated-{tag}-cal-passed']=str(sum(r['calibration_passes'] for r in cells))
   out[f'generated-{tag}-cal-cells']=str(len(cells))
 return out
def table(lines,columns,header,rows,caption,label):
 lines.extend([r'\begin{table*}[t]',r'\centering\small',r'\sbox{\roundfivebox}{\begin{tabular}{@{}'+columns+r'@{}}',r'\toprule',header+r' \\',r'\midrule'])
 lines.extend(' & '.join(row)+r' \\' for row in rows)
 lines.extend([r'\bottomrule',r'\end{tabular}}',r'\ifdim\wd\roundfivebox>\textwidth\resizebox{\textwidth}{!}{\usebox{\roundfivebox}}\else\usebox{\roundfivebox}\fi',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table*}'])
def main():
 lines=['% Generated from validated reviewer follow-up JSON.',r'\newsavebox{\roundfivebox}']
 x=load('competence')
 if x:
  lines.extend([r'\section{Task competence and value readouts}',r'\label{app:round5-competence}',
   f'Table~\\ref{{tab:round5-competence}} compares both variables for {x["n_models"]} models. Layers are selected by neutral selectivity, averaging three probe seeds. Task accuracy and paired lure excess pool demonstration sets 7, 11 and 13; probes use set 7. The within-model difference holds overall task accuracy fixed but does not isolate a causal effect of the readout.',
   'We average the two variable readouts within each model before computing descriptive correlations. Table~\\ref{tab:round5-correlation} repeats them after omitting each model. Roles are not independent observations. With this small, heterogeneous panel, correlations cannot distinguish overall competence from the role of decodable information.'])
  rows=[[LABEL.get(r['model'],r['model']),r['role'],str(r['layer'])]+[f'{100*r[k]:.1f}' for k in ('task_accuracy','neutral_accuracy','misleading_accuracy','lure_excess')] for r in x['cells']]
  table(lines,'llrrrrr','Model & Role & Layer & Task & Neutral probe & Misleading probe & Excess',rows,
   'Level-3 arithmetic task and probe accuracies (\\%) and paired lure excess (points). Neutral task accuracy is shared by the two roles. All available complete model pairs are shown; below-floor models remain descriptive.','tab:round5-competence')
  columns=list(x['correlations'])
  rows=[['All models']+[number(x['correlations'][k]['pearson']) for k in columns]]
  rows += [[LABEL.get(m,m)]+[number(d[k]['pearson']) for k in columns] for m,d in x['leave_one_model_out'].items()]
  table(lines,'lrrr','Omitted model & Task--excess & Probe--excess & Task--probe',rows,
   'Descriptive Pearson correlations of model means; Spearman values and within-model variable differences are in the saved summary. No significance or causal claim is based on these coefficients.','tab:round5-correlation')
  rows=[[LABEL.get(r['model'],r['model'])]+[f'{100*r[k]:+.1f}' for k in ('neutral_accuracy','misleading_accuracy','lure_excess')] for r in x['within_model_v2_minus_v1']]
  table(lines,'lrrr','Model & Neutral probe & Misleading probe & Excess',rows,
   'Intermediate minus queried variable, in percentage points. Overall neutral task accuracy is identical within each model. These differences are descriptive comparisons between variable positions.','tab:round5-within')
 x=load('generated')
 if x:
  lines.extend([r'\section{Gold-trained transfer to generated chains}',r'\label{app:round5-generated}',
   'We read the state before the first numeric assignment to the target in each saved generated chain. Expression operands are not counted as completed value writes; tokens that include any part of the supplied value are excluded. A fixed sample of 200 matched sets is selected by a hash independent of outcomes, shared across models, variables and demonstration sets. All additional lure-writing chains and their neutral twins are evaluated separately.',
   'Neutral gold-chain probes use 10,000 separate training problems, three probe seeds and the original neutral-selectivity layer. Table~\\ref{tab:round5-generated} tests transfer to held-out neutral generated chains. A high-accuracy interpretation requires at least 90\\% neutral calibration in that demonstration set. Cases below that threshold yield descriptive readouts only. The augmentation is selected by observed errors and cannot estimate their population rate. Appendix~\\ref{app:round6-generated} separately trains probes on generated neutral chains and tests the same held-out prefixes.'])
  for model in sorted({r['model'] for r in x['cells']}):
   cells=[r for r in x['cells'] if r['model']==model]
   passed=sum(r['calibration_passes'] for r in cells)
   lo=min(r['neutral_calibration_ci95'][0] for r in cells if r['neutral_calibration_ci95'])
   hi=max(r['neutral_calibration_ci95'][0] for r in cells if r['neutral_calibration_ci95'])
   lines.append(f'{LABEL[model]} meets the neutral transfer threshold in {passed} of {len(cells)} cells, with calibration accuracy from {100*lo:.1f} to {100*hi:.1f}\\%.')
  lines.append('Lure-writing cases in the competent model are few; error-conditioned readouts cannot establish a general mechanism from that subset. Failed neutral calibration prevents a strong generated-chain interpretation for the exception model.')
  rows=[]
  for r in x['cells']:
   rows.append([LABEL[r['model']],str(r['demo_seed']),r['role'],f"{r['n_primary_neutral_valid']}/{r['n_primary_neutral']}",ci(r['neutral_calibration_ci95']),ci(r['primary_lure_probe_ci95']),str(r['n_augmented_lure_valid']),ci(r['augmented_lure_probe_ci95'])])
  table(lines,'lrrlllll','Model & Set & Role & Neutral coverage & Calibration & Sample lure probe & Extra lure $n$ & Extra lure probe',rows,
   'Generated-chain true-digit accuracy (\\%) [95\\% interval]. Sample lure probe conditions on lure writes in the fixed sample; extra lure cases exclude that sample. Dashes indicate no eligible cases. Saved summaries record all missing or excluded boundaries and calibration decisions.','tab:round5-generated')
 (ROOT/'paper/tables/round5_followups.tex').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
