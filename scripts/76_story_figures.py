#!/usr/bin/env python
"""Self-contained main figures: fixed arithmetic, response format, and readout calibration.

CPU only: python scripts/76_story_figures.py
Uses saved summaries, never models/tokenizers. Existing detailed figures remain in the appendix.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cueconf.config import PROJECT_ROOT, RESULTS_DIR

RED, BLUE, GREEN, GREY = '#B5321F', '#2B4A9E', '#157A55', '#5B6470'
MODELS = [('llama32-3b', 'Llama-3.2-3B'), ('llama32-3b-it', 'Llama-3.2-3B Instruct'),
          ('llama31-8b', 'Llama-3.1-8B'), ('llama31-8b-it', 'Llama-3.1-8B Instruct'),
          ('gemma3-4b-it', 'Gemma-3-4B Instruct'), ('gemma3-12b-it', 'Gemma-3-12B Instruct'),
          ('olmo2-1b-it', 'OLMo-2-1B Instruct*'), ('olmo2-7b-it', 'OLMo-2-7B Instruct')]


def save(fig, name, payload):
    dest = PROJECT_ROOT / 'paper' / 'figures'; dest.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(dest/f'{name}.{ext}', dpi=180)
    (dest/f'{name}_data.json').write_text(json.dumps(payload, indent=2)+'\n')
    print('wrote', dest/f'{name}.pdf')


def behavior(plt):
    sweep = json.loads((RESULTS_DIR/'summary/seed_sweep_L3.json').read_text())
    eq = json.loads((RESULTS_DIR/'summary/equivalence.json').read_text())
    cells = {k: v for k,v in eq['cells'].items() if '/cot/lure_excess@' in k}
    count = sum(c['equivalent'] for c in cells.values())
    assert len(cells) == eq['summary']['cot_cells'] and count == eq['summary']['cot_equivalent']
    fig = plt.figure(figsize=(7.2, 2.8))
    ax = fig.add_axes([.28, .20, .70, .61])
    payload = {'level3_queried_variable': {}, 'cot_equivalence': cells}
    for y, (key, label) in enumerate(MODELS):
        for regime, offset, color, marker, legend in [
            ('direct', -.13, RED, 'o', 'Answer only'),
            ('cot', .13, BLUE, 's', 'Written calculation')]:
            c = sweep[f'{key}/{regime}']['contrasts']['lure_excess@v1']
            mean = 100*c['pooled_mean']; lo, hi = [100*x for x in c['ci95']]
            ax.errorbar(mean, y+offset, xerr=[[mean-lo], [hi-mean]], fmt=marker,
                        color=color, ms=4, capsize=2, label=legend if y == 0 else None)
            payload['level3_queried_variable'].setdefault(key, {})[regime] = c
    ax.set_yticks(range(8), [lab for _,lab in MODELS]); ax.invert_yaxis()
    ax.tick_params(axis='y', length=0, labelsize=8)
    ax.set_xlim(-1, 6); ax.set_xticks([0, 2, 4, 6]); ax.axvline(0, color='.65', lw=.8)
    ax.grid(axis='x', alpha=.15)
    ax.set_xlabel('Added misleading-digit answers (points)', fontsize=8.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.57, .99),
               ncol=2, frameon=False, fontsize=9)
    ax.set_title('Queried variable, two-operation problems', fontsize=10, pad=7)
    save(fig, 'story_behavior', payload)


def readouts(plt):
    data = json.loads((RESULTS_DIR/'summary/round6_generated.json').read_text())
    assert data['validated']
    fig = plt.figure(figsize=(7.2, 1.9))
    ax = fig.add_axes([.25, .29, .72, .49])
    ax.set_title('Generated calculations, ordinary names', fontsize=10, pad=8)
    payload = {}
    for y, (key, label, color) in enumerate([
        ('llama32-3b', 'Llama-3.2-3B', BLUE), ('olmo2-1b-it', 'OLMo-2-1B Instruct', RED)]):
        triples = [(role, seed, cell['neutral_calibration_ci95'])
                   for role, values in data['models'][key].items()
                   for seed, cell in values['cells'].items()]
        payload[key] = triples
        for j, (role, seed, triple) in enumerate(triples):
            mean, lo, hi = [100*x for x in triple]
            ax.errorbar(mean, y+(j-2.5)*.09, xerr=[[mean-lo], [hi-mean]], fmt='o',
                        ms=3.5, color=color, capsize=2, lw=1)
    ax.set_yticks([0,1], ['Llama-3.2-3B','OLMo-2-1B Instruct']); ax.invert_yaxis()
    ax.set_ylim(1.5, -.5); ax.set_xlim(0, 102); ax.set_xticks([0, 50, 90, 100]); ax.tick_params(axis='y', length=0)
    ax.axvline(90, color=GREY, ls='--', lw=1)
    ax.text(88, -.39, '90% minimum', ha='right', fontsize=8, color=GREY)
    ax.grid(axis='x', alpha=.15); ax.set_xlabel('Correct digit predicted (%)', fontsize=9)
    save(fig, 'story_readouts', payload)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    behavior(plt); readouts(plt)


if __name__ == '__main__':
    main()
