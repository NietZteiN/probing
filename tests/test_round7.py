import importlib.util
from pathlib import Path

import pytest

from cueconf.generator import sample_single
from cueconf.round7 import format_output, parse_factorial, parse_final, verified_calculation


def test_factorial_never_supplies_test_values_in_program_and_unnamed_steps_remove_names():
    x = next(sample_single(3, 1, 71009, 'neutral'))
    named = format_output(x, 'names_equations')
    unnamed = format_output(x, 'no_names_equations')
    assert all(name in named and name not in unnamed for name in x.names.values())
    assert parse_factorial(named) == x.answer == parse_factorial(unnamed)
    assert '+' not in format_output(x, 'names_values')
    assert '-' not in format_output(x, 'names_values')


def test_fixed_calculation_rejects_correct_final_answer_after_wrong_intermediate():
    x = next(sample_single(3, 1, 71009, 'neutral'))
    assert verified_calculation(x, x.cot+'\n\n') == x.cot
    role = next(role for role in x.values if role != x.query)
    bad = x.cot.replace(f'{x.names[role]}={x.values[role]}', f'{x.names[role]}={(x.values[role]+1)%10}')
    assert verified_calculation(x, bad) is None
    assert verified_calculation(x, x.direct) is None


@pytest.mark.parametrize('text', ['Here is 3', 'Answer: 3', 'new problem 3', ''])
def test_final_parser_rejects_nonanswer_prefixes(text):
    assert parse_final(text) is None


def test_final_parser_keeps_multidigit_errors_and_factorial_stops_before_new_problem():
    assert parse_final(' 15\n') == 15
    assert parse_factorial('2, 3; Answer: 3\n\nAnswer: 8') == 3


def load_analysis_script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_prompting_subtracts_neutral_digit_changes_in_both_conditions():
    import pandas as pd
    analysis = load_analysis_script('85_paired_prompting.py')
    direct = pd.DataFrame([
        {'set_id':'one','group':'neutral','pred':2,'lure':None,'correct':True},
        {'set_id':'one','group':'incongruent@v1','pred':8,'lure':8,'correct':False}])
    cot = direct.copy()
    cot.loc[cot.group=='neutral','pred'] = 8
    cot.loc[cot.group=='neutral','correct'] = False
    d,c = analysis.matched_effects(direct,cot,'v1')
    assert d.excess.iloc[0] == 1
    assert c.excess.iloc[0] == 0
    assert (c-d).excess.iloc[0] == -1
    assert (c-d).misleading_accuracy.iloc[0] == 0


def test_control_contrasts_reject_changed_cohorts_and_gate_on_accuracy():
    collector = load_analysis_script('87_collect_response_controls.py')
    base = [{'seed':7,'set_id':str(i),'program_key':str(i),'excess':1,
             'neutral_correct':1} for i in range(3)]
    changed = [{**r,'excess':0,'neutral_correct':0} for r in base]
    comparison = collector.contrast(changed,base)
    assert comparison['change_ci95'] == [-1,-1,-1]
    assert comparison['competence_matched'] is False
    with pytest.raises(ValueError,match='eligible pairs'):
        collector.contrast(changed[:-1],base)
