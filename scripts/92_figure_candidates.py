#!/usr/bin/env python
"""Plot three optional combined-paper figures from saved summaries, CPU only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / 'paper_combined/figure_candidates'
MODELS = [('olmo2-7b-it', 'OLMo-2-7B-Instruct'),
          ('llama32-3b-it', 'Llama-3.2-3B-Instruct'),
          ('llama31-8b-it', 'Llama-3.1-8B-Instruct')]
RED, GREEN, BLUE, GRAY = '#B5321F', '#157A55', '#2B4A9E', '#65717C'
SOURCES, POINTS = {}, {}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'font.family': 'DejaVu Sans'})


def load(repo, filename):
    path = WORKSPACE / repo / 'results/summary' / (filename + '.json')
    SOURCES[str(path.relative_to(WORKSPACE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = json.loads(path.read_text())
    if 'validated' in data:
        assert data['validated'], path
    return data


def interval(ax, values, y, color, marker='o', label=None):
    mean, lo, hi = np.array(values, dtype=float) * 100
    assert np.isfinite([mean, lo, hi]).all() and lo <= mean <= hi
    ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], color=color,
                fmt=marker, ms=5, capsize=2.5, lw=1.2, label=label, zorder=3)
    return mean


def clean_axis(ax, ticks, limits):
    ax.set_xlim(*limits)
    ax.set_xticks(ticks)
    ax.grid(axis='x', color='#e8e8e8', lw=.6, zorder=0)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.axvline(0, color='#aaa', lw=.7, zorder=1)


def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight', pad_inches=.08)
    fig.savefig(OUT / (name + '.png'), bbox_inches='tight', pad_inches=.08, dpi=220)
    plt.close(fig)


def operation_controls():
    inference = load('codecue', 'cell_inference')
    controls = load('codecue', 'round5_formats')
    labels = ['Values only\nv = 2', 'Values + neutral text\nv = 2 [notes]',
              'Numeric elaboration\nv = 2 + 0 = 2', 'Source operation + value\nv = len(xs) = 2']
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.9), sharey=True)
    fig.subplots_adjust(left=.30, right=.98, top=.79, bottom=.22, wspace=.23)
    fig.suptitle('Code task: compute list length with v = len(xs)', y=.99, fontsize=10)
    for ax, (model, title) in zip(axes, MODELS[:2]):
        rows = []
        for regime in ('trace', 'trace_expr'):
            cell = inference['cells'][f'L5/{model}/{regime}/len-to-sum']
            assert cell['n'] == 855 and cell['n_sets'] == 285
            rows.append([cell['excess'], cell['lo'], cell['hi']])
        values = [rows[0], controls['models'][model]['neutral_annotation']['excess_ci95'],
                  controls['models'][model]['numeric_elaboration']['excess_ci95'], rows[1]]
        POINTS.setdefault('operation_controls', {})[model] = values
        for y, vals, color in zip(range(4), values, [RED, GRAY, BLUE, GREEN]):
            mean = interval(ax, vals, y, color)
            ax.annotate(f'{mean:.1f}', (mean, y), xytext=(0, 9),
                        textcoords='offset points', ha='center', fontsize=8, color=color)
        clean_axis(ax, [0, 20, 40, 60], (-3, 68))
        ax.set_title(title, fontsize=9, pad=15)
        ax.set_yticks(range(4), labels)
        ax.set_ylim(3.5, -.6)
    fig.supxlabel('Name-suggested wrong writes, misleading − ordinary (points)',
                  fontsize=9, y=.07)
    save(fig, 'operation_controls')


def fresh_names():
    data = load('codecue', 'round6_identifier_replication')
    names = data['selected_names']
    assert names == ['sum_of_values', 'sum_of_items', 'sum_of_elements']
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 3.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.24, right=.99, top=.82, bottom=.20, wspace=.19, hspace=.63)
    fig.suptitle('List-length task · fresh names on 200 new computations', y=.995, fontsize=10)
    for ax, (model, title) in zip(axes.flat, MODELS + [('codegemma-7b-it', 'CodeGemma-7B-Instruct')]):
        row = data['models'][model]
        assert row['n_programs'] == 200 and row['n_names'] == 3
        POINTS.setdefault('fresh_names', {})[model] = {}
        for y, name in enumerate(names):
            trace, expression = [row[regime]['by_name'][name] for regime in ('trace', 'trace_expr')]
            POINTS['fresh_names'][model][name] = {'values': trace, 'expression': expression}
            ax.plot([100*trace[0], 100*expression[0]], [y, y], color='#c7c7c7', lw=1)
            interval(ax, trace, y-.09, RED, 'o', 'Values only' if y == 0 else None)
            interval(ax, expression, y+.09, GREEN, 's', 'Source operation + value' if y == 0 else None)
        clean_axis(ax, [0, 25, 50, 75, 100], (-4, 104))
        ax.set_title(title, fontsize=9, pad=8)
        ax.set_yticks(range(3), names, fontfamily='DejaVu Sans Mono', fontsize=8)
        ax.set_ylim(2.45, -.45)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.60, .945),
               ncol=2, frameon=False, fontsize=9)
    fig.supxlabel('Name-suggested wrong writes, misleading − ordinary (points)',
                  fontsize=9, y=.07)
    save(fig, 'fresh_names')


def correct_calculation():
    data = load('probing', 'round7_response_controls')
    row = data['models']['olmo2-1b-it']['final_query']['v1']
    labels = ['Query omits variable name', 'Query repeats variable name']
    keys = ['repeat0_named0', 'repeat1_named0']
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.4), sharey=True)
    fig.subplots_adjust(left=.30, right=.98, top=.67, bottom=.33, wspace=.32)
    fig.suptitle('After fixed, completely correct calculations\nOLMo-2-1B-Instruct · 368 pairs / 205 computations',
                 y=.995, fontsize=10)
    POINTS['correct_calculation'] = {}
    for y, key in enumerate(keys):
        cell = row['cells'][key]
        assert cell['n_pairs'] == 368 and cell['n_computations'] == 205
        POINTS['correct_calculation'][key] = {
            field: cell[field] for field in ('excess_ci95', 'neutral_correct_ci95')}
        for ax, field, color in zip(axes, ('excess_ci95', 'neutral_correct_ci95'), (RED, BLUE)):
            mean = interval(ax, cell[field], y, color)
            ax.annotate(f'{mean:.1f}', (mean, y), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=8, color=color)
    axes[0].set_title('Added name-suggested answers', fontsize=9, pad=13)
    axes[1].set_title('Ordinary-name accuracy', fontsize=9, pad=13)
    clean_axis(axes[0], [0, 5, 10, 15], (-1, 18))
    clean_axis(axes[1], [70, 80, 90, 100], (68, 102))
    axes[0].set_xlabel('Misleading − ordinary (points)')
    axes[1].set_xlabel('Correct answers (%)')
    for ax in axes:
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(1.45, -.65)
    fig.text(.30, .015, 'Both formats output a digit. Ordinary-name accuracy also changes.', fontsize=8)
    save(fig, 'correct_calculation')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    operation_controls()
    fresh_names()
    correct_calculation()
    (OUT / 'provenance.json').write_text(json.dumps({
        'sources_sha256': SOURCES, 'plotted_intervals_proportions': POINTS,
        'intervals': 'Saved pointwise 95% computation-cluster bootstrap intervals; no new inference.',
    }, indent=2) + '\n')
    print('Wrote three candidate figures (PDF and PNG), with source hashes and plotted values.')


if __name__ == '__main__':
    main()
