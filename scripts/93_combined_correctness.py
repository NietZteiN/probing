#!/usr/bin/env python
"""Audit correctness on the existing matched code-format cohort; CPU only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parent / 'codecue'
sys.path.insert(0, str(CODE / 'src'))
from codecue.config import DATA_DIR, OUT_DIR
from codecue.stats import bootstrap_ci

MODELS = ('olmo2-7b-it', 'llama32-3b-it', 'llama31-8b-it')
SOURCE_HASHES = {}


def read(path):
    prefix, relative = ('codecue', path.relative_to(CODE)) if path.is_relative_to(CODE) else ('codecue_outputs', path.relative_to(OUT_DIR))
    SOURCE_HASHES[f'{prefix}/{relative}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return [json.loads(line) for line in path.open()]


def main():
    programs = {r['id']: r for r in read(DATA_DIR / 'L5/test_sets.jsonl')}
    inference_path = CODE / 'results/summary/cell_inference.json'
    SOURCE_HASHES['codecue/results/summary/cell_inference.json'] = hashlib.sha256(inference_path.read_bytes()).hexdigest()
    inference = json.loads(inference_path.read_text())['cells']
    comparison_path = CODE / 'results/summary/round7_prompt_comparison.json'
    SOURCE_HASHES['codecue/results/summary/round7_prompt_comparison.json'] = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    comparison = json.loads(comparison_path.read_text())
    assert comparison['validated']
    result = {'validated': True, 'source_files_sha256': SOURCE_HASHES,
              'cohort': 'Original level-5 length-to-sum, first assignment, 285 matched programs x three demonstration sets.',
              'bootstrap': '4000 draws over matched program IDs, keeping names and demonstrations together; pointwise 95% intervals.',
              'models': {}}
    for model in MODELS:
        records = {}
        for regime in ('trace', 'trace_expr'):
            rows = []
            for seed in (7, 11, 13):
                suffix = '' if seed == 7 else f'_s{seed}'
                directory = OUT_DIR / 'runs' / model / 'L5' / (regime + suffix)
                neutral = {r['set_id']: r for r in read(directory / 'neutral/behavior.jsonl')}
                for r in read(directory / 'incongruent@v1/behavior.jsonl'):
                    x = programs[r['id']]
                    if x['stmts'][0]['op'] != 'len' or r['lure_name'] not in ('total', 'sum_all', 'acc'):
                        continue
                    n = neutral[r['set_id']]
                    assert x['target'] == r['target'] == 'v1'
                    assert x['values']['v1'] == r['values']['v1'] == n['values']['v1']
                    assert x['answer'] == r['answer'] == n['answer']
                    assert r['lure'] != x['values']['v1']
                    assert r['value_written'] == r['values_written'].get('v1')
                    assert r['wrote_true_value'] == (r['value_written'] == x['values']['v1'])
                    assert r['correct'] == (r['pred'] == x['answer'])
                    assert n['correct'] == (n['pred'] == x['answer'])
                    rows.append({'set_id': r['set_id'], 'seed': seed,
                        'computation': json.dumps([x['stmts'], x['xs']], sort_keys=True),
                        'neutral_step_correct': int(n['values_written'].get('v1') == x['values']['v1']),
                        'misleading_step_correct': int(r['value_written'] == x['values']['v1']),
                        'neutral_final_correct': int(n['correct']),
                        'misleading_final_correct': int(r['correct']),
                        'excess': int(r['value_written'] == r['lure']) - int(n['values_written'].get('v1') == r['lure'])})
            assert len(rows) == 855 and len({r['set_id'] for r in rows}) == 285
            assert len({r['computation'] for r in rows}) == 285
            assert len({(r['set_id'], r['seed']) for r in rows}) == 855
            cell = inference[f'L5/{model}/{regime}/len-to-sum']
            assert np.isclose(np.mean([r['excess'] for r in rows]), cell['excess'])
            records[regime] = rows
        assert [(r['set_id'], r['seed']) for r in records['trace']] == [(r['set_id'], r['seed']) for r in records['trace_expr']]
        assert sum(r['misleading_step_correct'] for r in records['trace']) == comparison['models'][model]['pooled']['correct_write']['n']
        summary = {}
        fields = ('neutral_step_correct', 'misleading_step_correct',
                  'neutral_final_correct', 'misleading_final_correct')
        for regime, rows in records.items():
            clusters = [r['set_id'] for r in rows]
            summary[regime] = {'n_pairs': 855, 'n_computations': 285,
                'correct_counts': {field: sum(r[field] for r in rows) for field in fields},
                **{field + '_ci95': list(bootstrap_ci([r[field] for r in rows], clusters, n_boot=4000)) for field in fields}}
        summary['expression_minus_values'] = {field + '_ci95': list(bootstrap_ci(
            [b[field] - a[field] for a, b in zip(records['trace'], records['trace_expr'])],
            [r['set_id'] for r in records['trace']], n_boot=4000)) for field in fields}
        result['models'][model] = summary
    destination = ROOT / 'paper_combined/format_correctness.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    for model, data in result['models'].items():
        print(model, {regime: data[regime]['correct_counts'] for regime in ('trace', 'trace_expr')})


if __name__ == '__main__':
    main()
