"""Variable-name vocabularies.

NUMBER_WORDS[d] is the English number word for digit d; it is the only source of lures, so a
lure is always a value the probe can express (rule 3: the class range is D = {0..9}).

NEUTRAL_CANDIDATES is deliberately larger than needed. Rule 4 (token-count matching) is decided
per tokenizer by `tokcheck.py`, which writes the surviving words to data/words/<model>.json;
the generator reads that file when it exists and falls back to the candidates otherwise (only
for tests and dry runs). A word appears in four contexts in our prompts -- at line start, after
", ", after "+" and after "-" -- and has to be a single token in every one of them.
"""
from __future__ import annotations

import string

NUMBER_WORDS: tuple[str, ...] = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
)
WORD_TO_DIGIT = {w: i for i, w in enumerate(NUMBER_WORDS)}

LETTERS: tuple[str, ...] = tuple(string.ascii_uppercase)

# Concrete, common, non-numeric, non-mathematical, no ordinal/quantity meaning. Words with a
# quantity sense (pair, dozen, half, twin, solo) and math words (sum, add) are excluded on
# purpose; so are words that are also SI prefixes or units (kilo, gram, inch).
NEUTRAL_CANDIDATES: tuple[str, ...] = (
    "pen", "cup", "box", "dog", "cat", "hat", "map", "key", "jar", "bag", "car", "bus", "fox",
    "owl", "pig", "cow", "ant", "bee", "egg", "ink", "jam", "log", "net", "oak", "pan", "rug",
    "sun", "tea", "van", "web", "hut", "fig", "gem", "ham", "kit", "lid", "mud", "nut", "pot",
    "rod", "sky", "toy", "wax", "yak", "fan", "cap", "bed", "bin", "rag", "saw", "tub", "bell",
    "boat", "book", "cake", "coat", "desk", "door", "duck", "farm", "fish", "fork", "frog",
    "goat", "hill", "lamp", "leaf", "lake", "milk", "moon", "nest", "park", "pear", "ring",
    "road", "rock", "roof", "rope", "salt", "seed", "ship", "shoe", "soap", "sock", "song",
    "soup", "star", "tent", "tree", "wall", "wolf", "wool", "yard", "apple", "bread", "chair",
    "cloud", "grape", "horse", "house", "lemon", "mouse", "piano", "river", "shirt", "table",
    "tiger", "train", "truck", "water", "wheel", "zebra",
)

# Words that must never be neutral names: anything a reader could take as a quantity.
FORBIDDEN_NEUTRAL = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "ten",
    "pair", "dozen", "half", "twin", "solo", "sum", "add", "plus", "minus", "none", "all",
    "first", "second", "third", "last", "many", "few", "some", "double", "triple", "unit",
}
assert not FORBIDDEN_NEUTRAL & set(NEUTRAL_CANDIDATES)


# ---------------------------------------------------------------------------------------------
# A second, stronger cue (E38): identifier-style names, the way programmers actually write them.
# `q4` puts the misleading number INSIDE the identifier instead of spelling it as an English word.
#
# TOKENIZATION. Every tokenizer in the panel splits a two-character identifier into stem and
# suffix, so a digit-suffixed name is two tokens. To keep twins aligned the neutral control must
# split the same way, which most letters do NOT: `xz`, `vw` and `vz` merge into one token. Only
# the stem `q` with the suffixes f, g and j splits identically to q0-q9 in all six models and in
# every context of our template, checked 2026-09-14. Hence these exact values.
IDENT_STEM = "q"
IDENT_NUMBER_NAMES: tuple[str, ...] = tuple(f"{IDENT_STEM}{d}" for d in range(10))
IDENT_NEUTRAL_NAMES: tuple[str, ...] = ("qf", "qg", "qj")


class NameScheme:
    """What a condition's names are drawn from. `word` is the paper's main scheme (English number
    words against common nouns); `ident` is the identifier scheme above."""

    def __init__(self, key: str, number_names: tuple[str, ...], neutral_pool: tuple[str, ...]):
        self.key, self.number_names, self.neutral_pool = key, number_names, neutral_pool
        self.to_digit = {w: i for i, w in enumerate(number_names)}


WORD_SCHEME = NameScheme("word", NUMBER_WORDS, NEUTRAL_CANDIDATES)
IDENT_SCHEME = NameScheme("ident", IDENT_NUMBER_NAMES, IDENT_NEUTRAL_NAMES)
SCHEMES = {"word": WORD_SCHEME, "ident": IDENT_SCHEME}
