"""Task generator: Kudo et al. (2026) levels, naming conditions, matched twins.

An INSTANCE is one problem in one naming condition. A MATCHED SET is the same arithmetic under
every condition (rule 6): identical equations, operators and digits, differing only in the
surface names. Every comparison in the paper is within matched sets.

Levels follow Kudo et al., Table 1 (roles v1, v2, v3 are variables in left-to-right order):

    L1  v1 = d ± v2, v2 = d              ; v1 = ?     1 step,  1 stack
    L2  v1 = d ± d,  v2 = d ± v1         ; v2 = ?     2 steps, 0 stack (in order)
    L3  v1 = d ± v2, v2 = d ± d          ; v1 = ?     2 steps, 1 stack   <- main level
    L4  L3 + distractor v3 = d ± d        ; v1 = ?     2 steps, 1 stack, 1 distractor
    L5  v1 = d ± v2, v2 = d ± v3, v3 = d ± d ; v1 = ?  3 steps, 2 stack

All digits and all variable values are in D = {0..9} (Kudo's data constraint), so every value a
probe must express is a single digit and every lure is a legal class.

Conditions (plan §4.2):
    letter       single capital letters, Kudo's original format
    neutral      single-token non-number words                     (main comparison point)
    congruent    the TARGET variable is named by the number word of its own true value
    incongruent  the TARGET variable is named by a number word that is NOT its value (lure)
    irrelevant   (L4 only) the DISTRACTOR is named by a number word; queried chain is neutral
    neutral_alt  neutral with a different neutral word in the target slot  (patching control)
    incongruent_alt  incongruent with a different lure                    (patching control)

The target is either the intermediate variable (v2 in L3) or the queried variable (v1 in L3);
plan rule 5. Exactly one variable per instance carries a number word (rule 1).

Generation rules enforced here and tested in tests/test_generator.py:
    R1  exactly one number-word name per congruent/incongruent/irrelevant instance
    R2  the lure differs from every variable value AND from every digit in the problem
        (stronger than the plan's wording: a lure that equals an operand digit, e.g.
        `two=2+3`, would have a second reason to be decodable)
    R3  the lure is a digit in 0..9 (it is, by construction: names come from NUMBER_WORDS)
    R6  twins are byte-identical after renaming
    plus Kudo's equation-level disjointness between probe-train and test splits, implemented as
    a HOLDOUT of digit expressions: the ~200 strings `d op d` are split by seed into a train
    pool and a test pool (`expression_split`), so an expression seen while training a probe
    never occurs in a test instance. Instances are unique at the rendered-text level (names
    included), as in Kudo et al.; arithmetic may repeat across matched sets under different
    names, which the cluster bootstrap over sets accounts for.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .words import LETTERS, NEUTRAL_CANDIDATES, NUMBER_WORDS, SCHEMES, WORD_SCHEME

DIGITS = range(10)
OPS = ("+", "-")

# role -> (operand spec) where an operand is either "d" (a fresh digit) or another role.
# The query role is listed separately. Order = surface order in the input.
LEVELS: dict[int, dict] = {
    1: {"eqs": [("v1", ("d", "v2")), ("v2", ("d",))], "query": "v1", "steps": 1, "distractor": None},
    2: {"eqs": [("v1", ("d", "d")), ("v2", ("d", "v1"))], "query": "v2", "steps": 2, "distractor": None},
    3: {"eqs": [("v1", ("d", "v2")), ("v2", ("d", "d"))], "query": "v1", "steps": 2, "distractor": None},
    4: {"eqs": [("v1", ("d", "v2")), ("v2", ("d", "d")), ("v3", ("d", "d"))], "query": "v1", "steps": 2, "distractor": "v3"},
    5: {"eqs": [("v1", ("d", "v2")), ("v2", ("d", "v3")), ("v3", ("d", "d"))], "query": "v1", "steps": 3, "distractor": None},
}

CONDITIONS = ("letter", "neutral", "congruent", "incongruent", "irrelevant", "neutral_alt", "incongruent_alt")
NUMBER_WORD_CONDITIONS = ("congruent", "incongruent", "irrelevant", "incongruent_alt")


@dataclass(frozen=True)
class Equation:
    lhs: str                 # role, e.g. "v2"
    op: str | None           # "+", "-" or None for a bare assignment
    args: tuple[str, ...]    # each arg is a digit string or a role


@dataclass
class Arithmetic:
    """A problem with roles instead of names. Everything a matched set shares."""
    level: int
    eqs: list[Equation]
    query: str
    values: dict[str, int]              # role -> value

    @property
    def digits(self) -> set[int]:
        return {int(a) for e in self.eqs for a in e.args if a.isdigit()}

    def key(self) -> str:
        """Equation-level identity used for train/test disjointness (Kudo footnote 3)."""
        return json.dumps([[e.lhs, e.op, list(e.args)] for e in self.eqs])

    def expressions(self) -> set[str]:
        """The digit expressions (digit op digit) in this problem, e.g. {'2+3'} (no spaces)."""
        out = set()
        for e in self.eqs:
            if e.op and all(a.isdigit() for a in e.args):
                out.add(f"{e.args[0]}{e.op}{e.args[1]}")
        return out


@dataclass
class Instance:
    id: str
    set_id: str
    level: int
    condition: str
    names: dict[str, str]                 # role -> surface name
    values: dict[str, int]                # role -> value
    query: str                            # role
    answer: int
    target: str | None                    # role carrying the number word, if any
    lure: int | None                      # the number word's value when it is not the true value
    lure_distance: int | None
    input: str
    cot: str
    direct: str
    eqs: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ----------------------------------------------------------------------------- arithmetic

def _eval(op: str | None, vals: Sequence[int]) -> int:
    if op is None:
        return vals[0]
    if op == "+":
        return vals[0] + vals[1]
    if op == "-":
        return vals[0] - vals[1]
    raise ValueError(op)


def _eval_str(a: int, op: str, b: int) -> int:
    return _eval(op, [a, b])


ALL_EXPRESSIONS: tuple[str, ...] = tuple(
    f"{a}{op}{b}" for a in DIGITS for op in OPS for b in DIGITS if 0 <= _eval_str(a, op, b) <= 9
)


def expression_split(seed: int = 44, test_frac: float = 0.3) -> tuple[set[str], set[str]]:
    """(train expressions, test expressions): a disjoint split of every valid `d op d`."""
    rng = random.Random(seed)
    ex = list(ALL_EXPRESSIONS)
    rng.shuffle(ex)
    n_test = int(round(test_frac * len(ex)))
    return set(ex[n_test:]), set(ex[:n_test])


def sample_arithmetic(level: int, rng: random.Random, max_tries: int = 100_000,
                      allowed_exprs: set[str] | None = None) -> Arithmetic:
    """Rejection-sample a problem whose values all fall in 0..9 (Kudo's constraint) and whose
    digit expressions all lie in `allowed_exprs` (train/test holdout)."""
    spec = LEVELS[level]
    for _ in range(max_tries):
        eqs: list[Equation] = []
        for lhs, arg_spec in spec["eqs"]:
            args = tuple(str(rng.choice(DIGITS)) if a == "d" else a for a in arg_spec)
            op = rng.choice(OPS) if len(args) == 2 else None
            eqs.append(Equation(lhs, op, args))
        values = _resolve(eqs)
        if values is None or not all(0 <= v <= 9 for v in values.values()):
            continue
        ar = Arithmetic(level, eqs, spec["query"], values)
        if allowed_exprs is not None and not ar.expressions() <= allowed_exprs:
            continue
        return ar
    raise RuntimeError(f"could not sample a level-{level} problem in {max_tries} tries")


def _resolve(eqs: list[Equation]) -> dict[str, int] | None:
    by_lhs = {e.lhs: e for e in eqs}
    values: dict[str, int] = {}

    def val(role: str, depth: int = 0) -> int:
        if depth > 10:
            raise RecursionError
        if role in values:
            return values[role]
        e = by_lhs[role]
        vals = [int(a) if a.isdigit() else val(a, depth + 1) for a in e.args]
        values[role] = _eval(e.op, vals)
        return values[role]

    try:
        for e in eqs:
            val(e.lhs)
    except RecursionError:
        return None
    return values


# ----------------------------------------------------------------------------- rendering

def render_eq(e: Equation, names: dict[str, str], subst: dict[str, int] | None = None) -> str:
    """`pen=1 + cup`; with subst, roles in subst are replaced by their values (`pen=1 + 5`).

    SPACING (decided 2026-09-12 from the Llama-3 tokenizer): binary operators carry a space on
    each side and `=` carries none. Without the space, `-` merges with the following word
    (`1-two` -> `1`, `-two`; `1-pen` -> `1`, `-p`, `en`), which breaks rule 4 for every name.
    With it, every name is `two` at line start and ` two` elsewhere, both single tokens. Kudo
    et al.'s paper renders equations with spaces as well (their Table 1); the exact byte format
    of their preprocessor was not reproduced, so absolute positions are not comparable to their
    figures, only the equation-level ones (t*_eq).
    """
    parts = []
    for a in e.args:
        if a.isdigit():
            parts.append(a)
        elif subst and a in subst:
            parts.append(str(subst[a]))
        else:
            parts.append(names[a])
    rhs = parts[0] if e.op is None else f"{parts[0]} {e.op} {parts[1]}"
    return f"{names[e.lhs]}={rhs}"


def render_input(ar: Arithmetic, names: dict[str, str]) -> str:
    return ", ".join(render_eq(e, names) for e in ar.eqs) + f"; {names[ar.query]}=?"


def cot_steps(ar: Arithmetic, names: dict[str, str]) -> list[tuple[str, str, str]]:
    """Kudo et al. Table 1 chain as (kind, role, text) steps; kinds: restate | substitute | value.

    L3: pen=1 + cup, cup=2 + 3, cup=5, pen=1 + cup, pen=1 + 5, pen=6
    L1: pen=1 + cup, cup=2, pen=1 + cup, pen=1 + 2, pen=3
    Distractors (L4) never appear in the chain, as in the paper. A bare assignment `cup=2` is
    restated once and that restatement IS its value step (kind "value").
    """
    by_lhs = {e.lhs: e for e in ar.eqs}
    steps: list[tuple[str, str, str]] = []

    def resolve(role: str) -> None:
        e = by_lhs[role]
        deps = [a for a in e.args if not a.isdigit()]
        if e.op is None:
            steps.append(("value", role, render_eq(e, names)))
            return
        steps.append(("restate", role, render_eq(e, names)))
        for d in deps:
            resolve(d)
        if deps:
            steps.append(("restate", role, render_eq(e, names)))
            steps.append(("substitute", role, render_eq(e, names, subst={d: ar.values[d] for d in deps})))
        steps.append(("value", role, f"{names[role]}={ar.values[role]}"))

    resolve(ar.query)
    return steps


def render_cot(ar: Arithmetic, names: dict[str, str]) -> str:
    return ", ".join(t for _, _, t in cot_steps(ar, names))


def render_direct(ar: Arithmetic, names: dict[str, str]) -> str:
    return f"{names[ar.query]}={ar.values[ar.query]}"


# ----------------------------------------------------------------------------- naming

def _roles(ar: Arithmetic) -> list[str]:
    return [e.lhs for e in ar.eqs]


def neutral_names(ar: Arithmetic, rng: random.Random, pool: Sequence[str]) -> dict[str, str]:
    words = rng.sample(list(pool), len(ar.eqs))
    return dict(zip(_roles(ar), words))


def letter_names(ar: Arithmetic, rng: random.Random) -> dict[str, str]:
    return dict(zip(_roles(ar), rng.sample(list(LETTERS), len(ar.eqs))))


def legal_lures(ar: Arithmetic, target: str) -> list[int]:
    """R2: a lure is a digit that is neither any variable's value nor any operand digit."""
    banned = set(ar.values.values()) | ar.digits
    return [d for d in DIGITS if d not in banned]


# ----------------------------------------------------------------------------- matched sets

def make_set(ar: Arithmetic, set_id: str, rng: random.Random, pool: Sequence[str],
             targets: Iterable[str] | None = None, scheme=WORD_SCHEME) -> list[Instance]:
    """All conditions for one arithmetic. `targets` defaults to every non-distractor role.
    `scheme` decides what a number-carrying name looks like: an English number word (the paper's
    main condition) or an identifier such as `q4` (E38)."""
    spec = LEVELS[ar.level]
    roles = _roles(ar)
    chain_roles = [r for r in roles if r != spec["distractor"]]
    targets = list(targets) if targets is not None else chain_roles
    base = neutral_names(ar, rng, pool)             # the neutral twin; all others rename ONE slot
    unused = [w for w in pool if w not in base.values()]
    alt_word = rng.choice(unused)
    out: list[Instance] = []

    def inst(cond: str, names: dict[str, str], target: str | None, lure: int | None, suffix: str) -> Instance:
        true = ar.values[target] if target else None
        return Instance(
            id=f"{set_id}-{suffix}", set_id=set_id, level=ar.level, condition=cond,
            names=dict(names), values=dict(ar.values), query=ar.query, answer=ar.values[ar.query],
            target=target, lure=lure,
            lure_distance=(abs(lure - true) if lure is not None else None),
            input=render_input(ar, names), cot=render_cot(ar, names), direct=render_direct(ar, names),
            eqs=[asdict(e) for e in ar.eqs],
        )

    out.append(inst("letter", letter_names(ar, rng), None, None, "letter"))
    out.append(inst("neutral", base, None, None, "neutral"))
    for t in targets:
        cong = dict(base); cong[t] = scheme.number_names[ar.values[t]]
        out.append(inst("congruent", cong, t, None, f"cong@{t}"))
        lures = legal_lures(ar, t)
        if len(lures) < 2:
            raise ValueError(f"{set_id}: fewer than two legal lures for {t}; resample")
        l1, l2 = rng.sample(lures, 2)
        inc = dict(base); inc[t] = scheme.number_names[l1]
        out.append(inst("incongruent", inc, t, l1, f"inc@{t}"))
        inc2 = dict(base); inc2[t] = scheme.number_names[l2]
        out.append(inst("incongruent_alt", inc2, t, l2, f"incalt@{t}"))
        nalt = dict(base); nalt[t] = alt_word
        out.append(inst("neutral_alt", nalt, t, None, f"nalt@{t}"))
    if spec["distractor"]:
        d = spec["distractor"]
        lures = legal_lures(ar, d)
        irr = dict(base); irr[d] = scheme.number_names[rng.choice(lures)]
        out.append(inst("irrelevant", irr, d, scheme.to_digit[irr[d]], f"irr@{d}"))
    return out


def irr_lure(names: dict[str, str], role: str) -> int:
    from .words import WORD_TO_DIGIT
    return WORD_TO_DIGIT[names[role]]


def sample_sets(level: int, n: int, seed: int, pool: Sequence[str] = NEUTRAL_CANDIDATES,
                allowed_exprs: set[str] | None = None, targets: Iterable[str] | None = None,
                scheme=WORD_SCHEME) -> Iterator[list[Instance]]:
    """n matched sets whose digit expressions lie in `allowed_exprs` (default: the test pool of
    `expression_split()`), unique at the rendered-input level."""
    rng = random.Random(seed)
    if allowed_exprs is None:
        allowed_exprs = expression_split()[1]
    seen: set[str] = set()
    made = tries = 0
    while made < n:
        tries += 1
        if tries > 100 * n:
            raise RuntimeError("generator stalled: constraints too tight for this level")
        ar = sample_arithmetic(level, rng, allowed_exprs=allowed_exprs)
        try:
            s = make_set(ar, f"L{level}-{seed}-{made:05d}", rng, pool, targets, scheme=scheme)
        except ValueError:
            continue
        neutral = next(x for x in s if x.condition == "neutral")
        if neutral.input in seen:
            continue
        seen.add(neutral.input)
        made += 1
        yield s


def sample_single(level: int, n: int, seed: int, condition: str, pool: Sequence[str] = NEUTRAL_CANDIDATES,
                  allowed_exprs: set[str] | None = None) -> Iterator[Instance]:
    """Probe-training instances in ONE condition (neutral by default; letter for the Kudo
    replication), expressions from `allowed_exprs` (default: the train pool)."""
    rng = random.Random(seed)
    if allowed_exprs is None:
        allowed_exprs = expression_split()[0]
    seen: set[str] = set()
    made = 0
    while made < n:
        ar = sample_arithmetic(level, rng, allowed_exprs=allowed_exprs)
        names = letter_names(ar, rng) if condition == "letter" else neutral_names(ar, rng, pool)
        text = render_input(ar, names)
        if text in seen:
            continue
        seen.add(text)
        yield Instance(
            id=f"L{level}-{seed}-train-{made:05d}", set_id=f"L{level}-{seed}-train-{made:05d}",
            level=level, condition=condition, names=names, values=dict(ar.values), query=ar.query,
            answer=ar.values[ar.query], target=None, lure=None, lure_distance=None,
            input=text, cot=render_cot(ar, names), direct=render_direct(ar, names),
            eqs=[asdict(e) for e in ar.eqs],
        )
        made += 1


def instance_arithmetic(inst: Instance) -> Arithmetic:
    eqs = [Equation(e["lhs"], e["op"], tuple(e["args"])) for e in inst.eqs]
    return Arithmetic(inst.level, eqs, inst.query, dict(inst.values))


def write_jsonl(path: Path, instances: Iterable[Instance]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for x in instances:
            f.write(x.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[Instance]:
    out = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out.append(Instance(**d))
    return out


def content_hash(instances: Iterable[Instance]) -> str:
    h = hashlib.sha256()
    for x in instances:
        h.update(x.input.encode()); h.update(b"\0"); h.update(x.cot.encode()); h.update(b"\n")
    return h.hexdigest()[:16]
