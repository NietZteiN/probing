#!/usr/bin/env python
"""Prepare frozen cohorts on CPU, then run both controls in single-GPU workers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cueconf.config import DATA_DIR, OUT_DIR, model_entry
from cueconf.generator import Instance, read_jsonl
from cueconf.prompts import make_demos, build_prompt
from cueconf.round6 import computation_key
from cueconf.round7 import (FORMATS, digest, factorial_prompt, final_prompt, matched_heads,
                           parse_factorial, parse_final, verified_calculation)

BASE = OUT_DIR / 'round7_controls'
MODELS = ('llama32-3b', 'llama31-8b', 'olmo2-1b-it')
SEEDS = (7, 11, 13)
COUNTS = (3, 8, 16)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def frozen(path, data):
    if path.exists() and json.loads(path.read_text()) != data:
        raise ValueError(f'frozen source changed: {path}')
    write(path, data)


def prepare():
    instances = read_jsonl(DATA_DIR / 'L3/test_sets.jsonl')
    by_id = {x.id: x for x in instances}
    by_set = {}
    for x in instances:
        group = f'{x.condition}@{x.target}' if x.target else x.condition
        by_set.setdefault(x.set_id, {})[group] = x
    demos = [x for seed in SEEDS for x in make_demos(3, 'word', seed=seed, n=16)]
    forbidden = {computation_key(x) for x in demos}
    selected = []
    for set_id in sorted(by_set, key=lambda value: digest('round7-primary:' + value)):
        group = by_set[set_id]
        key = computation_key(group['neutral'])
        if key in forbidden:
            continue
        forbidden.add(key)
        selected.append(group)
        if len(selected) == 200:
            break
    if len(selected) != 200:
        raise ValueError('insufficient independent primary computations')
    rows = [asdict(group[label]) for group in selected
            for label in ('neutral', 'incongruent@v1', 'incongruent@v2')]
    all_test = {computation_key(x) for x in instances}
    calibration = []
    for x in sorted(read_jsonl(DATA_DIR / 'L3/probe_train_neutral.jsonl'),
                    key=lambda value: digest('round7-calibration:' + value.id)):
        key = computation_key(x)
        if key in all_test or key in forbidden:
            continue
        forbidden.add(key)
        calibration.append(asdict(x))
        if len(calibration) == 100:
            break
    if len(calibration) != 100:
        raise ValueError('insufficient independent calibration computations')
    cohort = {'level': 3, 'n_computations': 200, 'primary': rows, 'calibration': calibration,
              'models': list(MODELS), 'seeds': list(SEEDS), 'example_counts': list(COUNTS),
              'formats': list(FORMATS), 'protocol_sha256': digest((ROOT/'docs/ROUND7_PROTOCOL.md').read_text()),
              'data_sha256': hashlib.sha256((DATA_DIR/'L3/test_sets.jsonl').read_bytes()).hexdigest()}
    frozen(BASE / 'cohorts.json', cohort)
    for model in MODELS:
        eligibility = {'model': model, 'source_sha256': {}, 'groups': {}}
        for seed in SEEDS:
            regime = 'cot' if seed == 7 else f'cot_s{seed}'
            source = {}
            for group in ('neutral', 'incongruent@v1', 'incongruent@v2'):
                path = OUT_DIR / 'runs' / model / 'L3' / regime / group / 'behavior.jsonl'
                eligibility['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                source[group] = {row['set_id']: row for row in map(json.loads, path.read_text().splitlines())}
            for role in ('v1', 'v2'):
                seen, accepted = set(), []
                for set_id in sorted(by_set, key=lambda value: digest('round7-final:' + value)):
                    group = by_set[set_id]
                    key = computation_key(group['neutral'])
                    if key in seen or key in {computation_key(x) for x in demos}:
                        continue
                    pair = []
                    for condition in ('neutral', f'incongruent@{role}'):
                        old = source[condition].get(set_id)
                        if old is None:
                            break
                        x = by_id[old['id']]
                        calculation = verified_calculation(x, old['generation'])
                        if calculation is None:
                            break
                        pair.append({'instance': asdict(x), 'calculation': calculation})
                    if len(pair) == 2:
                        seen.add(key)
                        accepted.append({'set_id': set_id, 'program_key': key, 'pair': pair})
                        if len(accepted) == 200:
                            break
                eligibility['groups'][f's{seed}/{role}'] = accepted
        frozen(BASE / model / 'final_cohort.json', eligibility)
        print(model, 'eligible final-query pairs', {k: len(v) for k,v in eligibility['groups'].items()}, flush=True)
    print('Frozen format and generated-calculation cohorts', flush=True)


def infer(tok, model, prompts, metadata, path, experiment, batch_size, budget):
    from cueconf.runner import generate_free
    signature = digest(json.dumps({'prompts': prompts, 'metadata': metadata, 'budget': budget}, sort_keys=True))
    if path.exists():
        data = json.loads(path.read_text())
        if not data['complete'] or data['signature'] != signature:
            raise ValueError(f'continuation cache mismatch: {path}')
        return data['records']
    records = []
    start = time.monotonic()
    parser = parse_factorial if experiment == 'formats' else parse_final
    for offset in range(0, len(prompts), 64):
        chunk = prompts[offset:offset+64]
        generations = generate_free(tok, model, chunk, budget, batch_size, stop_at_newline=True)
        if len(generations) != len(chunk):
            raise ValueError('incomplete generation')
        for prompt, generation, row in zip(chunk, generations, metadata[offset:offset+64]):
            prediction = parser(generation)
            records.append({**row, 'prediction': prediction, 'generation': generation,
                'correct': prediction == row['answer'], 'parsed': prediction is not None,
                'prompt_sha256': digest(prompt),
                'prompt_tokens': len(tok(prompt, add_special_tokens=True)['input_ids']),
                'generated_tokens': len(tok(generation, add_special_tokens=False)['input_ids'])})
        print(path.name, len(records), '/', len(prompts), f'{time.monotonic()-start:.1f}s', flush=True)
    write(path, {'complete': True, 'signature': signature, 'greedy': True, 'max_new_tokens': budget,
                 'experiment': experiment, 'seconds': time.monotonic()-start, 'records': records,
                 'job_id': os.environ.get('SLURM_JOB_ID')})
    return records


def run(model_key, batch_size):
    from cueconf.runner import load_model
    import torch, transformers
    start = time.monotonic()
    cohort = json.loads((BASE/'cohorts.json').read_text())
    entry = model_entry(model_key)
    tok, model = load_model(entry['hf_id'])
    primary = [Instance(**row) for row in cohort['primary']]
    calibration = [Instance(**row) for row in cohort['calibration']]
    output = BASE/model_key
    selected, passing = 16, False
    calibration_report = {}
    for count in COUNTS:
        demos = make_demos(3, 'word', seed=7, n=count)
        heads, audit = matched_heads(tok, demos)
        accuracy = {}
        for style in FORMATS:
            prompts = [factorial_prompt(x, heads[style]) for x in calibration]
            meta = [{'id': x.id, 'set_id': x.set_id, 'program_key': computation_key(x),
                     'condition': x.condition, 'target': None, 'lure': None, 'answer': x.answer,
                     'format': style, 'seed': 7} for x in calibration]
            rows = infer(tok, model, prompts, meta, output/'calibration'/f'k{count}_{style}.json',
                         'formats', batch_size, 128)
            accuracy[style] = sum(row['correct'] for row in rows)/len(rows)
        passed = min(accuracy.values()) >= .90 and max(accuracy.values())-min(accuracy.values()) <= .0200001
        calibration_report[str(count)] = {'accuracy': accuracy, 'pass': passed, 'token_matching': audit}
        print('calibration', model_key, count, accuracy, 'pass', passed, flush=True)
        if passed:
            selected, passing = count, True
            break
    write(output/'selection.json', {'count': selected, 'calibration_pass': passing,
         'calibration': calibration_report, 'selection_uses_test_outcomes': False})
    for seed in SEEDS:
        heads, audit = matched_heads(tok, make_demos(3, 'word', seed=seed, n=selected))
        write(output/'formats'/f's{seed}_token_matching.json', audit)
        for style in FORMATS:
            prompts = [factorial_prompt(x, heads[style]) for x in primary]
            meta = [{'id': x.id, 'set_id': x.set_id, 'program_key': computation_key(x),
                     'condition': x.condition, 'target': x.target, 'lure': x.lure, 'answer': x.answer,
                     'format': style, 'seed': seed} for x in primary]
            # Renaming alone must preserve token counts within each selected problem.
            for index in range(0, len(prompts), 3):
                if len({len(tok(p, add_special_tokens=True)['input_ids']) for p in prompts[index:index+3]}) != 1:
                    raise ValueError('matched names change prompt token positions')
            infer(tok, model, prompts, meta, output/'formats'/f's{seed}_{style}.json',
                  'formats', batch_size, 128)
    final_cohort = json.loads((output/'final_cohort.json').read_text())
    for seed in SEEDS:
        regime = 'cot' if seed == 7 else f'cot_s{seed}'
        demos = make_demos(3, 'word', seed=seed)
        for role in ('v1','v2'):
            pairs = final_cohort['groups'][f's{seed}/{role}']
            for repeated in (False, True):
                for named in (False, True):
                    prompts, meta = [], []
                    for pair in pairs:
                        for item in pair['pair']:
                            x = Instance(**item['instance'])
                            head = build_prompt(x, demos, regime)[:-len(x.input)-1]
                            prompts.append(final_prompt(x, head, item['calculation'], repeated, named))
                            meta.append({'id': x.id, 'set_id': x.set_id, 'program_key': pair['program_key'],
                                'condition': x.condition, 'target': role, 'lure': pair['pair'][1]['instance']['lure'],
                                'answer': x.answer, 'seed': seed, 'repeat_name': repeated,
                                'named_output': named, 'calculation_sha256': digest(item['calculation'])})
                    path = output/'final_query'/f's{seed}_{role}_repeat{int(repeated)}_named{int(named)}.json'
                    infer(tok, model, prompts, meta, path, 'final_query', batch_size, 16)
    write(output/'complete.json', {'complete': True, 'model': model_key, 'hf_id': entry['hf_id'],
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'job_id': os.environ.get('SLURM_JOB_ID'), 'seconds': time.monotonic()-start,
        'finished_utc': datetime.now(timezone.utc).isoformat()})
    print('COMPLETE', model_key, flush=True)


def preflight():
    from transformers import AutoTokenizer
    cohort = json.loads((BASE/'cohorts.json').read_text())
    primary = [Instance(**row) for row in cohort['primary']]
    report = []
    for model_key in MODELS:
        tok = AutoTokenizer.from_pretrained(model_entry(model_key)['hf_id'], local_files_only=True)
        for seed in SEEDS:
            for count in COUNTS:
                heads, audit = matched_heads(tok, make_demos(3, 'word', seed=seed, n=count))
                for index in range(0, len(primary), 3):
                    lengths = [len(tok(factorial_prompt(x, head), add_special_tokens=True)['input_ids'])
                               for x in primary[index:index+3] for head in heads.values()]
                    if len(set(lengths)) != 1:
                        raise ValueError('full prompt lengths differ across names or formats')
                report.append({'model': model_key, 'seed': seed, 'count': count, **audit})
    write(ROOT/'log/round7_2026-10-09/preflight.json', {'passed': True, 'checks': report})
    print('All tokenizer and full-prompt matching checks passed', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--models', nargs='+', choices=MODELS)
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    if args.prepare:
        prepare()
    if args.preflight:
        preflight()
    if args.models:
        for model in args.models:
            run(model, args.batch_size)


if __name__ == '__main__':
    main()
