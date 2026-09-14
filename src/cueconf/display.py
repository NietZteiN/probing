"""Human-readable names for everything the figures and tables show.

Internal labels are terse because they index arrays (`cotpre@v2`, `in:v1`, `4:subs:v1`).
Nothing a reader sees should use them. One mapping, used by every figure script.
"""
from __future__ import annotations

import re

# roles -> the running example's names, so an axis reads like the problem does
EXAMPLE = {"v1": "pen", "v2": "cup", "v3": "box"}

ROLE_WORD = {"v1": "queried", "v2": "intermediate", "v3": "third"}


def position(label: str, role_names: dict[str, str] | None = None) -> str:
    """`cotpre@v2` -> 'before cup=5'. Falls back to the role word when no names are given."""
    n = role_names or EXAMPLE
    if label == "query":
        return "query '?'"
    if label == "anspre":
        return "before answer"
    if label == "ans":
        return "answer"
    if "@" not in label:
        return label
    kind, role = label.split("@")
    nm = n.get(role, role)
    return {"def": f"name '{nm}'", "use": f"'{nm}' used", "end": f"end of {nm}'s equation",
            "cotpre": f"before {nm}'s value", "cot": f"{nm}'s value"}.get(kind, label)


def segment(label: str, role_names: dict[str, str] | None = None) -> str:
    """`in:v1` -> \"pen's equation\"; `cot:4:substitute:v1` -> 'step 5: pen=1+5'."""
    n = role_names or EXAMPLE
    if label == "query":
        return "query line"
    if label.startswith("in:"):
        return f"{n.get(label[3:], label[3:])}'s equation"
    m = re.match(r"cot:(\d+):(\w+):(\w+)", label)
    if not m:
        return label
    k, kind, role = int(m.group(1)), m.group(2), m.group(3)
    nm = n.get(role, role)
    what = {"restate": f"restate {nm}", "substitute": f"substitute into {nm}", "value": f"write {nm}'s value"}[kind]
    return f"step {k + 1}: {what}"


def condition(label: str) -> str:
    """`incongruent@v2` -> 'incongruent (intermediate)'."""
    if "@" not in label:
        return label
    cond, role = label.split("@")
    pretty = {"incongruent_alt": "incongruent, other lure", "neutral_alt": "neutral, other word",
              "irrelevant": "irrelevant lure"}.get(cond, cond)
    return f"{pretty} ({ROLE_WORD.get(role, role)})"


MODEL = {"llama32-3b": "Llama-3.2-3B", "llama32-3b-it": "Llama-3.2-3B-Instruct",
         "llama31-8b": "Llama-3.1-8B", "llama31-8b-it": "Llama-3.1-8B-Instruct",
         "gemma3-4b-it": "Gemma-3-4B-it", "gemma3-12b-it": "Gemma-3-12B-it",
         "olmo2-1b-it": "OLMo-2-1B-Instruct", "olmo2-7b-it": "OLMo-2-7B-Instruct"}

REGIME = {"cot": "with chain of thought", "direct": "direct answer", "free": "free-form reasoning"}


def rcparams() -> dict:
    """Readable defaults: no 5-point tick labels, no matplotlib blue."""
    return {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
            "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 150,
            "axes.spines.top": False, "axes.spines.right": False}
