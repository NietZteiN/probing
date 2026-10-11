#!/usr/bin/env python
"""Render one verified pair of saved code generations; no model loading."""
from pathlib import Path
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PAPER = ROOT / 'paper_combined'
IDENTIFIER = 'L5-3-v1-00001-incongruent'
BLUE, RED, GREEN, GRAY = '#2B4A9E', '#B5321F', '#157A55', '#65717C'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    paths = {'program': WORKSPACE / 'codecue/data/L5/test_sets.jsonl'}
    outputs = Path(os.environ.get('CODECUE_OUT', f"/scratch/juno/{os.environ.get('USER', 'x')}/codecue"))
    run_root = outputs / 'runs/olmo2-7b-it/L5'
    for regime in ('trace', 'trace_expr'):
        paths[regime] = run_root / regime / 'incongruent@v1/behavior.jsonl'
    rows = {}
    for key, path in paths.items():
        matches = [json.loads(line) for line in path.open()
                   if json.loads(line)['id'] == IDENTIFIER]
        assert len(matches) == 1, (key, len(matches))
        rows[key] = matches[0]
    p, wrong, correct = [rows[k] for k in ('program', 'trace', 'trace_expr')]
    assert p['program'] == 'def f(xs):\n    sum_all = len(xs)\n    ww = sum_all + 3\n    qq = ww + 4\n    return qq\nf([5, 3])'
    assert p['values'] == wrong['values'] == correct['values'] == {'v1': 2, 'v2': 5, 'v3': 9}
    assert p['answer'] == wrong['answer'] == correct['answer'] == 9
    assert wrong['value_written'] == wrong['lure'] == 8 and wrong['pred'] == 15 and not wrong['correct']
    assert correct['value_written'] == 2 and correct['pred'] == 9 and correct['correct']
    # The saved example is a behavior illustration. No per-example probe prediction is asserted.
    fig = plt.figure(figsize=(7.2, 2.85))
    axes = [fig.add_axes(box) for box in [(.015,.17,.31,.76),(.355,.17,.29,.76),(.68,.17,.315,.76)]]
    for ax in axes:
        ax.axis('off')

    def code_line(ax, text, y, value_color=None, fontsize=8.1):
        advance = fontsize / 72 * .602 / (7.2 * ax.get_position().width)
        x = 0
        for chunk in re.split(r'(sum_all|(?<![\w])\d+(?![\w]))', text):
            color = BLUE if chunk == 'sum_all' else value_color if chunk.isdigit() and value_color else '#222222'
            ax.text(x, y, chunk, transform=ax.transAxes, family='DejaVu Sans Mono',
                    fontsize=fontsize, va='top', color=color)
            x += len(chunk) * advance
        return advance

    axes[0].text(0,1,'Program and input',weight='bold',fontsize=9,va='top')
    for j,line in enumerate(p['program'].splitlines()):
        code_line(axes[0],line,.85-j*.105)
    axes[0].text(0,.17,'Correct first value  2',fontsize=8.5,color=GREEN)
    axes[0].text(0,.055,'Correct final answer  9',fontsize=8.5,color=GREEN)
    marker = None
    for ax,title,row,color in [(axes[1],'Values-only\ndemonstrations',wrong,RED),
                               (axes[2],'Expression-and-value\ndemonstrations',correct,GREEN)]:
        ax.text(0,1,title,weight='bold',fontsize=9,va='top',linespacing=1.12)
        ax.text(0,.75,'Actual OLMo-2-7B output',fontsize=7.7,color=GRAY,va='top')
        assignments = row['generation'].splitlines()[0].strip().split(', ')
        assert len(assignments) == 3
        for j,assignment in enumerate(assignments):
            text = assignment + (',' if j < 2 else '')
            # Color only the reported value; operands retain ordinary text.
            prefix,value = text.rsplit(' = ',1)
            advance = code_line(ax,prefix+' = ',.58-j*.15)
            x = len(prefix+' = ') * advance
            ax.text(x,.58-j*.15,value,transform=ax.transAxes,family='DejaVu Sans Mono',
                    fontsize=8.1,va='top',color=color)
            if ax is axes[1] and j == 0:
                marker = (x-.5*advance,.615)
        ax.text(0,.055,f"Answer: {row['pred']}  ({'correct' if row['correct'] else 'wrong'})",
                fontsize=9,weight='bold',color=color)
    axes[1].annotate('Probe reads state',xy=marker,xycoords='axes fraction',
                     xytext=(.14,.66),textcoords='axes fraction',fontsize=7.5,
                     arrowprops={'arrowstyle':'->','lw':.8,'color':GRAY},color=GRAY)
    fig.text(.015,.09,'Misleading name',color=BLUE,fontsize=8)
    fig.text(.22,.09,'Incorrect value',color=RED,fontsize=8)
    fig.text(.425,.09,'Correct value',color=GREEN,fontsize=8)
    fig.text(.015,.015,"Probe reads at '=' before the first value; individual prediction not shown.",fontsize=7.7,color=GRAY)
    payload = {'id':IDENTIFIER,'model':'olmo2-7b-it','demonstration_seed':7,
               'program':p['program'],'correct_values':p['values'],'correct_answer':9,
               'name_suggested_value':8,'values_only_generation':wrong['generation'],
               'expression_and_value_generation':correct['generation'],
               'probe_prediction_shown':False,'display_changes':'Assignment line breaks only; comma separators retained.',
               'source_sha256':{('codecue/data/L5/test_sets.jsonl' if k=='program' else
                                f'codecue_outputs/runs/olmo2-7b-it/L5/{k}/incongruent@v1/behavior.jsonl'):
                               hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}}
    dest = PAPER / 'figures/shared_example'
    fig.savefig(dest.with_suffix('.pdf'),bbox_inches='tight',metadata={'CreationDate':None})
    fig.savefig(dest.with_suffix('.png'),bbox_inches='tight',dpi=200)
    dest.with_name('shared_example_data.json').write_text(json.dumps(payload,indent=2)+'\n')
    plt.close(fig)
    print('Verified and rendered saved example',IDENTIFIER)


if __name__ == '__main__':
    main()
