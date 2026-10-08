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
    fig = plt.figure(figsize=(7.2, 4.2))
    fig.text(.02, .97, 'Writing calculations reduces answers suggested by a misleading name',
             fontsize=11.5, weight='bold', va='top')
    fig.text(.02, .90, 'Only the variable name changes:', fontsize=10, weight='bold')
    fig.text(.02, .82, 'pen = 1 + cup\ncup = 2 + 3\npen = ?', family='monospace', fontsize=10,
             va='top', linespacing=1.2)
    fig.text(.27, .75, 'rename\npen → four', fontsize=9, ha='center', color=GREY)
    fig.text(.39, .82, 'four = 1 + cup\ncup = 2 + 3\nfour = ?', family='monospace', fontsize=10,
             va='top', linespacing=1.2)
    fig.text(.70, .83, 'Correct answer stays 6.', color=GREEN, fontsize=10, weight='bold')
    fig.text(.70, .76, 'Answering 4 follows the name.', color=RED, fontsize=9)
    fig.text(.70, .69, '4 is absent from the math.', fontsize=9)
    fig.text(.02, .64, '(a) Same problems, two requested response formats', weight='bold', fontsize=10)
    ax = fig.add_axes([.26, .22, .37, .31])
    payload = {'level3_queried_variable': {}, 'cot_equivalence': cells}
    for y, (key, label) in enumerate(MODELS):
        for regime, offset, color, marker, legend in [
            ('direct', -.13, RED, 'o', 'Final answer only'),
            ('cot', .13, BLUE, 's', 'Write calculations, then answer')]:
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
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.02,.60),
               ncol=2, frameon=False, fontsize=9, borderaxespad=0)
    fig.text(.68, .64, '(b) Across task depths\nand both variable roles', fontsize=10, weight='bold', va='top')
    fig.text(.68, .48, f'{count} of {len(cells)}', fontsize=21, color=BLUE, weight='bold')
    fig.text(.68, .42, 'With written calculations, the\nadded error rate is within\n2 points of zero.', fontsize=10, linespacing=1.4, va='top')
    fig.text(.68, .27, '90% intervals. These are errors\nmatching the wrong name,\nnot all incorrect answers.', fontsize=8.5, color=GREY, linespacing=1.3, va='top')
    fig.text(.02, .10, 'Added errors = renamed rate − ordinary-name rate. Plot: two-operation tasks; 95% intervals.', fontsize=8.5)
    fig.text(.02, .04, '* Below the planned 90% neutral-calculation accuracy floor; included as a scope check.', fontsize=8)
    save(fig, 'story_behavior', payload)


def readouts(plt):
    data = json.loads((RESULTS_DIR/'summary/round6_generated.json').read_text())
    assert data['validated']
    fig = plt.figure(figsize=(7.2, 2.7))
    fig.text(.02, .96, 'Predictors pass the accuracy check in Llama-3.2-3B, but fail in OLMo-2-1B',
             fontsize=12, weight='bold', va='top')
    fig.text(.02, .82, 'Separate predictor reads internal activity → predicts the variable’s correct digit.', fontsize=10)
    fig.text(.02, .68, 'Train and test on different ordinary-name problems, using calculations generated by the model.', fontsize=9)
    ax = fig.add_axes([.24, .24, .48, .33])
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
        passing = sum(c['calibration_pass'] for r in data['models'][key].values() for c in r['cells'].values())
        fig.text(.76, .49-y*.17, f'{passing}/6 meet the\n90% minimum', color=color,
                 fontsize=10, weight='bold', va='center')
    ax.set_yticks([0,1], ['Llama-3.2-3B','OLMo-2-1B Instruct']); ax.invert_yaxis()
    ax.set_ylim(1.5, -.5); ax.set_xlim(0, 102); ax.set_xticks([0, 50, 90, 100]); ax.tick_params(axis='y', length=0)
    ax.axvline(90, color=GREY, ls='--', lw=1)
    ax.text(88, -.39, '90% minimum', ha='right', fontsize=8, color=GREY)
    ax.grid(axis='x', alpha=.15); ax.set_xlabel('Correct digit predicted on ordinary-name problems (%)', fontsize=9)
    fig.text(.02, .025, 'Six points per model: 2 variables × 3 example sets; 95% intervals. Failed checks limit error-case interpretation.', fontsize=8.5)
    save(fig, 'story_readouts', payload)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    behavior(plt); readouts(plt)


if __name__ == '__main__':
    main()
