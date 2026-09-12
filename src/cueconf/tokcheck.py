"""Rule 4: every variable name must be ONE token in every context it appears in, per tokenizer.

Contexts in our layout: line start ("pen=..."), after ", " (", pen=..."), after " - " and
" + " as an operand, before "," and ";". Each context is tokenised with the word in place and
must have exactly the token count it would have if the word were one token and nothing merged:
the Llama-3 tokenizer merges "-" with a following word ("1-two" -> "1", "-two"), which is why
operators carry spaces in our format (generator.render_eq). The rendered template of one
instance is also tokenised end to end so a merged "=?" or ", " shows up in the report rather
than in a misaligned probe.

Run inside a SLURM job (the login node cannot load a tokenizer under its 8 GB cap):
    python scripts/11_tokenizer_check.py --model llama32-3b
Writes data/words/<model>.json = {"neutral": [...], "number_ok": bool, "number_ntok": {...}}.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .words import NEUTRAL_CANDIDATES, NUMBER_WORDS


def ntok(tokenizer, s: str) -> int:
    return len(tokenizer(s, add_special_tokens=False)["input_ids"])


# (context template, expected token count when w is ONE token and merges with nothing)
CONTEXTS = ((("{w}="), 2), ((", {w}="), 3), (("1 - {w},"), 4), (("1 + {w},"), 4), (("1 - {w};"), 4))


def word_ok(tokenizer, w: str) -> bool:
    """One token in every context, and the neighbours keep their own tokens (no merge)."""
    for tpl, n in CONTEXTS:
        ids = tokenizer(tpl.format(w=w), add_special_tokens=False)["input_ids"]
        pieces = [tokenizer.decode([i]) for i in ids]
        if len(ids) != n or sum(1 for p_ in pieces if p_.strip() == w) != 1:
            return False
    return True


def check_words(tokenizer, candidates: Sequence[str] = NEUTRAL_CANDIDATES) -> dict:
    number_ntok = {w: (ntok(tokenizer, w), ntok(tokenizer, " " + w), word_ok(tokenizer, w)) for w in NUMBER_WORDS}
    neutral = [w for w in candidates if word_ok(tokenizer, w)]
    rejected = [w for w in candidates if w not in neutral]
    return {
        "number_ntok": number_ntok,
        "number_ok": all(v[2] for v in number_ntok.values()),
        "neutral": neutral,
        "rejected": rejected,
    }


def template_report(tokenizer, text: str) -> list[tuple[int, str]]:
    enc = tokenizer(text, add_special_tokens=True)
    return [(i, tokenizer.decode([t])) for i, t in enumerate(enc["input_ids"])]


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
