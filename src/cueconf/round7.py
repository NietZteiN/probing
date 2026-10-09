"""Response-format and fixed-calculation controls; no model imports."""
from __future__ import annotations

import hashlib
import re

from .generator import Instance, instance_arithmetic, cot_steps

FORMATS = ('names_equations', 'no_names_equations', 'names_values', 'no_names_values')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def format_output(instance: Instance, style: str):
    if style not in FORMATS:
        raise ValueError('unknown factorial format')
    names = style.startswith('names_')
    equations = style.endswith('_equations')
    ar = instance_arithmetic(instance)
    by_role = {e.lhs: e for e in ar.eqs}
    # Value steps fix the dependency order independently of name spelling.
    roles = [role for kind, role, _ in cot_steps(ar, instance.names) if kind == 'value']
    steps = []
    for role in roles:
        equation = by_role[role]
        lhs = instance.names[role] + '=' if names else ''
        if equations and equation.op:
            args = [arg if arg.isdigit() else str(instance.values[arg]) for arg in equation.args]
            expression = f'{args[0]} {equation.op} {args[1]}='
        else:
            expression = ''
        steps.append(lhs + expression + str(instance.values[role]))
    return 'Steps: ' + ', '.join(steps) + f'; Answer: {instance.answer}'


def example_head(demos, style, notes=''):
    note = f'Notes:{notes}\n' if notes else ''
    return ''.join(f'{d.input}\n{note}{format_output(d, style)}\n\n' for d in demos)


def matched_heads(tok, demos):
    # Neutral filler is placed before every demonstrated output, never after its value.
    # All four heads include this line; filler counts vary to match the full head length.
    size = lambda text: len(tok(text, add_special_tokens=True)['input_ids'])
    heads, counts = {style: '' for style in FORMATS}, {style: [] for style in FORMATS}
    for demo in demos:
        target = max(size(example_head([demo], style, ' note')) for style in FORMATS) + 1
        for style in FORMATS:
            for count in range(1, 257):
                candidate = example_head([demo], style, ' note' * count)
                length = size(candidate)
                if length == target:
                    heads[style] += candidate
                    counts[style].append(count)
                    break
                if length > target:
                    raise ValueError('token-exact demonstration matching failed')
            else:
                raise ValueError('cannot match demonstration token lengths')
    assert len({size(head) for head in heads.values()}) == 1
    return heads, {'tokens': size(heads[FORMATS[0]]), 'filler': counts}


def factorial_prompt(instance, head):
    return head + instance.input + '\nSteps:'


def parse_factorial(generation):
    block = generation.split('\n\n', 1)[0]
    match = re.search(r'\bAnswer:\s*(-?\d+)\b', block)
    return int(match.group(1)) if match else None


def verified_calculation(instance, generation):
    # Strict, conservative eligibility: every restatement, substitution and value equals gold.
    block = generation.split('\n\n', 1)[0].strip()
    normalize = lambda text: re.sub(r'\s+', '', text)
    if normalize(block) != normalize(instance.cot):
        return None
    return block


def final_prompt(instance, head, calculation, repeat_name, named_output):
    query = instance.names[instance.query] + '=?' if repeat_name else 'the original requested value=?'
    output = instance.names[instance.query] + '=' if named_output else ''
    return (head + instance.input + '\n' + calculation
            + '\nReturn the value requested in the original problem.'
            + '\nFinal query: ' + query + '\nAnswer: ' + output)


def parse_final(generation):
    # The prompt already ends with the output marker. Reject prose or another problem first.
    match = re.match(r'\s*(-?\d+)\b', generation)
    return int(match.group(1)) if match else None
