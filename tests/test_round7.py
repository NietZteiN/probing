from dataclasses import replace

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
