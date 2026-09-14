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

REGIMES = ("cot", "direct", "free")
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
    """'cot' -> ('cot', 3); 'direct_k16' -> ('direct', 16); 'cot_s11' -> ('cot', 3) with demo seed 11
    (see demo_seed_of). Suffixes name a different prompt, so runs never share an output directory."""
    base = regime.split("_")[0]
    if base not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    k = N_DEMOS
    for part in regime.split("_")[1:]:
        if part.startswith("k"):
            k = int(part[1:])
    return base, k


def demo_seed_of(regime: str) -> int:
    """'cot_s11' -> 11; otherwise DEMO_SEED. A different seed draws different fixed demonstrations
    (E33: is a result an artefact of the three demos?)."""
    for part in regime.split("_")[1:]:
        if part.startswith("s"):
            return int(part[1:])
    return DEMO_SEED


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
    base = parse_regime(regime)[0]
    return inst.cot if base in ("cot", "free") else inst.direct


FREE_INSTRUCTION = ("Solve for the queried variable. Think step by step, then write the final line "
                    "as the variable name, an equals sign and the value.\n\n")


def build_prompt(inst: Instance, demos: Sequence[Instance], regime: str) -> str:
    """The free regime shows NO worked examples: only the instruction and the problem, so the
    model chooses its own reasoning format (E36). Its labelled positions therefore exist only in
    the prompt; the answer is parsed from free text."""
    if parse_regime(regime)[0] == "free":
        return FREE_INSTRUCTION + inst.input + "\n"
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
    segments: list[tuple[str, int, int]] = None  # (label, start, end) char spans: in:v1, in:v2, query, cot:0:restate:v1, ...

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
    segments: list[tuple[str, int, int]] = []
    for i, e in enumerate(ar.eqs):
        seg_start = cur
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
        segments.append((f"in:{e.lhs}", seg_start, cur))
        emit(", " if i < len(ar.eqs) - 1 else "; ")
    seg_start = cur
    at = emit(names[ar.query]); spans[ar.query].append((at, at + len(names[ar.query])))
    emit("=")
    pos["query"] = emit("?")
    segments.append(("query", seg_start, cur))
    emit("\n")
    assert "".join(buf) == inst.input + "\n", ("input re-render mismatch", "".join(buf), inst.input)
    assert cur == len(prompt)
    for r in names:
        pos[f"use@{r}"] = first_use.get(r)

    # ---- output
    for r in names:
        pos[f"cotpre@{r}"] = None
        pos[f"cot@{r}"] = None
    if parse_regime(regime)[0] in ("cot", "free"):
        steps = cot_steps(ar, names)
        obuf = []
        for k, (kind, role, text) in enumerate(steps):
            if k:
                emit(", ")
            at = emit(text)
            segments.append((f"cot:{k}:{kind}:{role}", at, at + len(text)))
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
        segments.append(("cot:0:value:" + ar.query, at, at + len(inst.direct)))
        spans[ar.query].append((at, at + len(names[ar.query])))
        eq_at = at + len(names[ar.query])
    pos["anspre"] = pos[f"cotpre@{ar.query}"] if parse_regime(regime)[0] in ("cot", "free") else eq_at
    pos["ans"] = pos["anspre"] + 1
    text = prompt[:start] + "".join(buf)
    assert text == prompt + output_of(inst, regime)
    assert text[pos["ans"]] == str(inst.answer)
    return Layout(prompt=prompt, output=output_of(inst, regime), positions=pos, name_spans=spans, instance_start=start, segments=segments)


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
    seg_tokens = {lab: sorted({cmap[c] for c in range(a, b) if cmap[c] >= 0}) for lab, a, b in (lay.segments or [])}
    inst_tok0 = cmap[lay.instance_start]
    return {
        "segment_tokens": seg_tokens,         # label -> token indices covered by that equation / step
        "instance_start_token": inst_tok0,
        "token_strings": [tokenizer.decode([t]) for t in enc["input_ids"][inst_tok0:]],
        "input_ids": enc["input_ids"],
        "n_prompt_tokens": n_prompt,
        "positions": tpos,
        "name_tokens": name_tokens,       # every token index carrying each role's name
        "name_ntok": name_ntok,           # distinct per-occurrence token counts (want [1])
        "n_tokens": len(enc["input_ids"]),
    }


def parse_answer_free(text: str, name: str) -> int | None:
    """Free-form generations: the LAST `name = <int>` anywhere in the text, allowing spaces,
    `**bold**`, a trailing period, and the model naming the variable in words ("pen is 6")."""
    import re
    pats = [rf"{re.escape(name)}\s*=\s*(-?\d+)", rf"{re.escape(name)}\s+is\s+(-?\d+)",
            rf"\*\*\s*{re.escape(name)}\s*=\s*(-?\d+)"]
    vals = [m.group(1) for p_ in pats for m in re.finditer(p_, text)]
    if not vals:
        m = list(re.finditer(r"(-?\d+)\s*$", text.strip()))
        return int(m[-1].group(1)) if m else None
    return int(vals[-1])


def parse_answer(text: str, name: str) -> int | None:
    """The model's answer to THIS problem: the last `name=<int>` written before the blank line
    that separates few-shot problems.

    Reading only the first line (the original rule) throws away correct answers whenever the
    model restates the problem before solving it, which it does for 5-13% of instances at some
    levels and at different rates per condition, so it biased the contrasts. Cutting at the
    blank line keeps the answer without picking up a fresh problem the model invents afterwards.
    """
    block = text.split("\n\n", 1)[0]
    val = None
    for piece in block.replace("\n", ",").split(","):
        piece = piece.strip()
        if piece.startswith(name + "="):
            rhs = piece[len(name) + 1:].strip()
            if rhs.lstrip("-").isdigit():
                val = int(rhs)
    return val
