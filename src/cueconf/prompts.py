"""Prompt rendering and probe positions.

Layout (one line per problem; the model continues after the final newline):

    <demo input>\\n<demo output>\\n\\n   x3
    <test input>\\n
    <output>                          generated, or forced (gold) for probing/patching

Three fixed demonstrations per (level, naming scheme), as in Kudo et al. (3 same-level shots,
the same three for every instance), so every instance of a (level, condition, regime) has the
same token layout and probes can be indexed by absolute position as in the paper. Letter
conditions get letter demos (Kudo's format exactly); every word condition gets the SAME neutral
demos, drawn from a reserved demo word pool that the test pool never uses.

Positions are computed on CHARACTERS from the instance structure, then mapped to tokens with
the tokenizer's offsets. Labels, per role r (v1, v2, ...) and globally:

    def@r      the name token where r is defined (lhs)         plan P1  (reference only)
    use@r      the name token where r is first used as an operand (None if never used)
    end@r      the last operand token of r's defining equation   plan P2  (first computable)
    query      the `?` token                                     plan P3
    cotpre@r   the `=` of r's value step in the chain            plan P4  (next token = value)
    cot@r      the value token of that step
    anspre     the `=` before the final answer                   plan P5
    ans        the final answer token

In the direct regime the chain positions do not exist (None) and anspre/ans are the only
output positions. cotpre@query == anspre in the CoT regime.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .generator import (Instance, cot_steps, instance_arithmetic, sample_single)
from .words import NEUTRAL_CANDIDATES

REGIMES = ("cot", "direct")
N_DEMOS = 3
DEMO_SEED = 7
N_DEMO_WORDS = 12          # reserved for demos, never used in probe-train or test instances


def split_pool(pool: Sequence[str] = NEUTRAL_CANDIDATES, seed: int = DEMO_SEED) -> tuple[list[str], list[str]]:
    """(demo words, instance words); deterministic given the pool."""
    rng = random.Random(seed)
    words = list(pool)
    rng.shuffle(words)
    return words[:N_DEMO_WORDS], words[N_DEMO_WORDS:]


def parse_regime(regime: str) -> tuple[str, int]:
    """'cot' -> ('cot', 3); 'direct_k16' -> ('direct', 16). The suffix names a demo count other
    than the default, so runs with different prompts never share an output directory."""
    base, _, k = regime.partition("_k")
    if base not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    return base, (int(k) if k else N_DEMOS)


def make_demos(level: int, scheme: str, pool: Sequence[str] = NEUTRAL_CANDIDATES, seed: int = DEMO_SEED,
               n: int = N_DEMOS) -> list[Instance]:
    """scheme: 'letter' or 'word'. Demo words come from the reserved pool; with n > its size the
    words repeat across demos, which is fine because demos are fixed and shared by every instance."""
    demo_words, _ = split_pool(pool, seed)
    cond = "letter" if scheme == "letter" else "neutral"
    return list(sample_single(level, n, seed, cond, pool=demo_words))


def scheme_of(condition: str) -> str:
    return "letter" if condition == "letter" else "word"


def output_of(inst: Instance, regime: str) -> str:
    return inst.cot if parse_regime(regime)[0] == "cot" else inst.direct


def build_prompt(inst: Instance, demos: Sequence[Instance], regime: str) -> str:
    head = "".join(f"{d.input}\n{output_of(d, regime)}\n\n" for d in demos)
    return head + inst.input + "\n"


def full_text(inst: Instance, demos: Sequence[Instance], regime: str) -> tuple[str, str]:
    """(prompt, gold output). Forced text is prompt + output."""
    return build_prompt(inst, demos, regime), output_of(inst, regime)


@dataclass
class Layout:
    prompt: str
    output: str
    positions: dict[str, int | None]        # label -> char index into prompt+output
    name_spans: dict[str, list[tuple[int, int]]]   # role -> [(start, end)] of every name occurrence
    instance_start: int                     # char offset where the test instance begins

    @property
    def text(self) -> str:
        return self.prompt + self.output


def layout(inst: Instance, demos: Sequence[Instance], regime: str) -> Layout:
    prompt = build_prompt(inst, demos, regime)
    start = len(prompt) - len(inst.input) - 1
    ar = instance_arithmetic(inst)
    names = inst.names
    pos: dict[str, int | None] = {}
    spans: dict[str, list[tuple[int, int]]] = {r: [] for r in names}

    # ---- input: rebuild it piece by piece, recording offsets; assert it matches inst.input
    buf = []
    cur = start

    def emit(s: str) -> int:
        nonlocal cur
        buf.append(s)
        at = cur
        cur += len(s)
        return at

    first_use: dict[str, int] = {}
    for i, e in enumerate(ar.eqs):
        at = emit(names[e.lhs]); spans[e.lhs].append((at, at + len(names[e.lhs])))
        pos[f"def@{e.lhs}"] = at
        emit("=")
        last_operand_at = None
        for j, a in enumerate(e.args):
            if j == 1:
                emit(f" {e.op} ")
            if a.isdigit():
                last_operand_at = emit(a)
            else:
                at = emit(names[a]); spans[a].append((at, at + len(names[a])))
                first_use.setdefault(a, at)
                last_operand_at = at + len(names[a]) - 1   # last char of the word
        pos[f"end@{e.lhs}"] = last_operand_at
        emit(", " if i < len(ar.eqs) - 1 else "; ")
    at = emit(names[ar.query]); spans[ar.query].append((at, at + len(names[ar.query])))
    emit("=")
    pos["query"] = emit("?")
    emit("\n")
    assert "".join(buf) == inst.input + "\n", ("input re-render mismatch", "".join(buf), inst.input)
    assert cur == len(prompt)
    for r in names:
        pos[f"use@{r}"] = first_use.get(r)

    # ---- output
    for r in names:
        pos[f"cotpre@{r}"] = None
        pos[f"cot@{r}"] = None
    if parse_regime(regime)[0] == "cot":
        steps = cot_steps(ar, names)
        obuf = []
        for k, (kind, role, text) in enumerate(steps):
            if k:
                emit(", ")
            at = emit(text)
            obuf.append(text)
            # every name occurrence inside this step (lhs and operand names)
            # scan tokens of the step: names are alphabetic runs
            idx = 0
            while idx < len(text):
                if text[idx].isalpha():
                    j = idx
                    while j < len(text) and text[j].isalpha():
                        j += 1
                    word = text[idx:j]
                    for rr, nm in names.items():
                        if nm == word:
                            spans[rr].append((at + idx, at + j))
                    idx = j
                else:
                    idx += 1
            if kind == "value":
                eq_at = at + len(names[role])
                assert text[len(names[role])] == "="
                pos[f"cotpre@{role}"] = eq_at
                pos[f"cot@{role}"] = eq_at + 1
        assert ", ".join(obuf) == inst.cot
    else:
        at = emit(inst.direct)
        spans[ar.query].append((at, at + len(names[ar.query])))
        eq_at = at + len(names[ar.query])
    pos["anspre"] = pos[f"cotpre@{ar.query}"] if parse_regime(regime)[0] == "cot" else eq_at
    pos["ans"] = pos["anspre"] + 1
    text = prompt[:start] + "".join(buf)
    assert text == prompt + output_of(inst, regime)
    assert text[pos["ans"]] == str(inst.answer)
    return Layout(prompt=prompt, output=output_of(inst, regime), positions=pos, name_spans=spans, instance_start=start)


def token_index_map(offsets: Sequence[tuple[int, int]]) -> list[int]:
    """char index -> token index, for offsets from a fast tokenizer (special tokens have (0,0))."""
    n = max((e for _, e in offsets), default=0)
    m = [-1] * n
    for t, (s, e) in enumerate(offsets):
        if e > s:
            for c in range(s, e):
                if m[c] == -1:
                    m[c] = t
    return m


def tokenize_layout(tokenizer, lay: Layout) -> dict:
    """Token positions for every label, plus name token counts (rule 4 check) and the ids."""
    enc = tokenizer(lay.text, return_offsets_mapping=True, add_special_tokens=True)
    cmap = token_index_map(enc["offset_mapping"])
    tpos = {k: (cmap[v] if v is not None else None) for k, v in lay.positions.items()}
    name_tokens = {r: sorted({cmap[c] for s, e in sp for c in range(s, e)}) for r, sp in lay.name_spans.items()}
    name_ntok = {r: (len({cmap[c] for c in range(s, e)}) for s, e in sp) for r, sp in lay.name_spans.items()}
    name_ntok = {r: sorted(set(v)) for r, v in name_ntok.items()}
    n_prompt = len(tokenizer(lay.prompt, add_special_tokens=True)["input_ids"])
    return {
        "input_ids": enc["input_ids"],
        "n_prompt_tokens": n_prompt,
        "positions": tpos,
        "name_tokens": name_tokens,       # every token index carrying each role's name
        "name_ntok": name_ntok,           # distinct per-occurrence token counts (want [1])
        "n_tokens": len(enc["input_ids"]),
    }


def parse_answer(text: str, name: str) -> int | None:
    """The value of `name=<digit>` LAST written on the first generated line, else None."""
    line = text.split("\n", 1)[0]
    val = None
    for piece in line.split(","):
        piece = piece.strip()
        if piece.startswith(name + "="):
            rhs = piece[len(name) + 1:].strip()
            if rhs.lstrip("-").isdigit():
                val = int(rhs)
    return val
