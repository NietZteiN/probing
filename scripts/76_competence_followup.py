#!/usr/bin/env python
"""Descriptive competence sensitivity; models, not roles, are the comparison units."""
from pathlib import Path
from statistics import mean
import importlib.util
import json
import sys
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cueconf.config import OUT_DIR, RESULTS_DIR
spec = importlib.util.spec_from_file_location('round4', ROOT / 'scripts/72_round4_completion.py')
r4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r4)


def correlation(rows, a, b):
    x, y = np.array([r[a] for r in rows]), np.array([r[b] for r in rows])
    if len(rows) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {'n_models':len(rows), 'pearson':None, 'spearman':None}
    return {'n_models':len(rows), 'pearson':float(np.corrcoef(x,y)[0,1]),
            'spearman':float(np.corrcoef(rankdata(x),rankdata(y))[0,1])}


def main():
    sweep = json.loads((RESULTS_DIR / 'summary/seed_sweep_L3.json').read_text())
    cells = []
    for directory in sorted((OUT_DIR / 'probes').iterdir()):
        model = directory.name
        if f'{model}/cot' not in sweep:
            continue
        for role in ('v1','v2'):
            primary = directory / f'L3/cot/train_neutral/{role}.json'
            suffix = '__r5_fp32' if model == 'gemma3-12b-it' else '__r4'
            extension = directory / f'L3/cot/train_neutral{suffix}/{role}.json'
            source = extension if extension.exists() else primary
            if not source.exists():
                continue
            if source == extension:
                data, train = r4.validate_probe(source, 10000, suffix)
            else:
                data = json.loads(source.read_text())
                train = json.loads((Path(data['train_dir']) / 'meta.json').read_text())
                if train['n'] != 10000 or {r['seed'] for r in data['results']} != {0,1,2}:
                    raise ValueError(f'incomplete primary probe {source}')
                train_sets = {x['set_id'] for x in train['instances']}
                test_meta = json.loads((Path(data['train_dir']).parent / 'neutral/meta.json').read_text())
                if train_sets & {x['set_id'] for x in test_meta['instances']}:
                    raise ValueError('overlapping arithmetic training sets')
            cell = r4.value_step(data, model, role)
            selected = [r for r in data['results'] if r['position']==f'cotpre@{role}' and r['layer']==cell['layer']]
            if len(selected)!=3 or {r['seed'] for r in selected}!={0,1,2} or not all(
                np.isfinite(r['eval'][f'incongruent@{role}']['margin_mean']) for r in selected):
                raise ValueError(f'invalid selected-layer arithmetic readout {source}')
            with np.load(source.with_suffix('.npz')) as arrays:
                for seed in (0,1,2):
                    margin=arrays[f'cotpre@{role}/L{cell["layer"]}/s{seed}/incongruent@{role}/margin']
                    if len(margin)!=len(data['test_ids'][f'incongruent@{role}']) or not np.isfinite(margin).all():
                        raise ValueError(f'invalid per-instance selected-layer margins {source}')
            behavior = sweep[f'{model}/cot']
            excess = behavior['contrasts'][f'lure_excess@{role}']
            cell.update(task_accuracy=behavior['groups']['neutral']['acc_mean'],
                        lure_excess=excess['pooled_mean'], lure_ci95=excess['ci95'],
                        source=str(source), below_competence_floor=behavior['groups']['neutral']['acc_mean']<.9)
            if not all(np.isfinite(cell[k]) for k in ('task_accuracy','neutral_accuracy','misleading_accuracy','lure_excess')):
                raise ValueError('non-finite competence readout')
            cells.append(cell)
    models = []
    differences = []
    for model in sorted({r['model'] for r in cells}):
        rs = [r for r in cells if r['model']==model]
        if {r['role'] for r in rs} != {'v1','v2'}:
            raise ValueError(f'incomplete roles for {model}')
        models.append({'model':model, **{k:mean(r[k] for r in rs) for k in
            ('task_accuracy','neutral_accuracy','misleading_accuracy','lure_excess')}})
        byrole = {r['role']:r for r in rs}
        differences.append({'model':model, 'task_accuracy':rs[0]['task_accuracy'],
            **{k:byrole['v2'][k]-byrole['v1'][k] for k in ('neutral_accuracy','misleading_accuracy','lure_excess')}})
    pairs = [('task_accuracy','lure_excess'),('misleading_accuracy','lure_excess'),('task_accuracy','misleading_accuracy')]
    describe = lambda rows: {f'{a}__{b}':correlation(rows,a,b) for a,b in pairs}
    result = {'validated':True,'n_models':len(models),'n_pairs':len(cells),'cells':cells,
        'models':models,'within_model_v2_minus_v1':differences,'correlations':describe(models),
        'leave_one_model_out':{r['model']:describe([x for x in models if x['model']!=r['model']]) for r in models},
        'interpretation':'Descriptive model-mean correlations only; no independent-role inference or causal attribution.'}
    (RESULTS_DIR / 'summary/round5_competence.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(9,2.8))
    for ax,(a,b) in zip(axes,pairs):
        for row in models:
            ax.scatter(100*row[a],100*row[b],s=25)
            ax.annotate(row['model'].replace('-it',''),(100*row[a],100*row[b]),fontsize=5,xytext=(2,2),textcoords='offset points')
        ax.set_xlabel(a.replace('_',' ')+' (%)',fontsize=8)
        ax.set_ylabel(b.replace('_',' ')+' (%)',fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    target=ROOT/'paper/figs/round5_competence.pdf';target.parent.mkdir(exist_ok=True)
    fig.savefig(target);plt.close(fig)
    print(f'Validated {len(cells)} role pairs across {len(models)} models',flush=True)

if __name__=='__main__': main()
