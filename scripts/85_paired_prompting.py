#!/usr/bin/env python
"""Paired CoT-minus-answer-only effects from the existing equivalence cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cueconf.stats import bootstrap_ci
from cueconf.generator import read_jsonl
from cueconf.round6 import computation_key


def matched_effects(direct, cot, role):
    frames = []
    for frame in (direct, cot):
        neutral = frame[frame.group == 'neutral'].set_index('set_id')
        misleading = frame[frame.group == f'incongruent@{role}'].set_index('set_id')
        if neutral.index.has_duplicates or misleading.index.has_duplicates:
            raise ValueError('duplicate matched sets')
        if set(neutral.index) != set(misleading.index):
            raise ValueError('incomplete naming pairs')
        misleading = misleading.loc[neutral.index]
        frames.append(pd.DataFrame({
            'excess': (misleading.pred.values == misleading.lure.values).astype(float)
                      - (neutral.pred.values == misleading.lure.values).astype(float),
            'neutral_accuracy': neutral.correct.astype(float).values,
            'misleading_accuracy': misleading.correct.astype(float).values,
            'interference': neutral.correct.astype(float).values - misleading.correct.astype(float).values,
        }, index=neutral.index))
    if set(frames[0].index) != set(frames[1].index):
        raise ValueError('prompting conditions use different problems')
    return frames[0], frames[1].loc[frames[0].index]


def main():
    equivalence = json.loads((ROOT / 'results/summary/equivalence.json').read_text())
    output = {'validated': True, 'direction': 'CoT minus answer-only',
              'confidence': .95, 'n_boot': 4000,
              'cluster': 'matched set across all three demonstration sets',
              'source': 'exact 44 CoT name-error cells from equivalence.json',
              'source_sha256': {}, 'cells': {}}
    computation_clusters = {}
    for key, reference in equivalence['cells'].items():
        level, model, regime, contrast = key.split('/')
        if regime != 'cot' or not contrast.startswith('lure_excess@'):
            continue
        role = contrast.split('@')[1]
        direct_frames, cot_frames = [], []
        for seed in reference['seeds']:
            pair = []
            for base in ('direct', 'cot'):
                tag = base if seed == 7 else f'{base}_s{seed}'
                path = ROOT / 'results/summary' / model / level / tag / 'behavior.csv'
                output['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                pair.append(pd.read_csv(path))
            d, c = matched_effects(*pair, role)
            direct_frames.append(d)
            cot_frames.append(c)
        d, c = pd.concat(direct_frames), pd.concat(cot_frames)
        if not np.isclose(c.excess.mean(), reference['mean']):
            raise ValueError(f'{key}: raw mean differs from released equivalence estimate')
        changes = c - d
        if level not in computation_clusters:
            computation_clusters[level] = {x.set_id: computation_key(x) for x in
                read_jsonl(ROOT / 'data' / level / 'test_sets.jsonl') if x.condition == 'neutral'}
        canonical = np.array([computation_clusters[level][set_id] for set_id in changes.index])
        intervals = {name: list(bootstrap_ci(changes[name].values, changes.index.values, n_boot=4000))
                     for name in changes}
        canonical_interval = list(bootstrap_ci(changes.excess.values, canonical, n_boot=4000))
        by_seed = {str(seed): float((cc.excess-dd.excess).mean())
                   for seed, dd, cc in zip(reference['seeds'], direct_frames, cot_frames)}
        consistent = all(value < 0 for value in by_seed.values())
        output['cells'][f'{level}/{model}/{role}'] = {
            'n': len(changes), 'n_sets': changes.index.nunique(), 'seeds': reference['seeds'],
            'direct_excess': float(d.excess.mean()), 'cot_excess': float(c.excess.mean()),
            'direct_neutral_accuracy': float(d.neutral_accuracy.mean()),
            'cot_neutral_accuracy': float(c.neutral_accuracy.mean()),
            'changes_ci95': intervals, 'excess_change_by_seed': by_seed,
            'canonical_excess_change_ci95': canonical_interval,
            'n_computations': len(set(canonical)),
            'reliable_reduction': consistent and intervals['excess'][2] < 0,
            'canonical_reliable_reduction': consistent and canonical_interval[2] < 0,
            'cot_equivalent': reference['equivalent'],
            'direct_equivalent': equivalence['cells'][key.replace('/cot/', '/direct/')]['equivalent'],
        }
    assert len(output['cells']) == 44
    output['counts'] = {
        'cells': len(output['cells']),
        'reliable_reductions': sum(c['reliable_reduction'] for c in output['cells'].values()),
        'negative_change_intervals': sum(c['changes_ci95']['excess'][2] < 0 for c in output['cells'].values()),
        'direct_equivalent': sum(c['direct_equivalent'] for c in output['cells'].values()),
        'cot_equivalent': sum(c['cot_equivalent'] for c in output['cells'].values()),
        'canonical_reliable_reductions': sum(c['canonical_reliable_reduction'] for c in output['cells'].values()),
    }
    destination = ROOT / 'results/summary/round7_paired_prompting.json'
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output['counts'], indent=2))


if __name__ == '__main__':
    main()
