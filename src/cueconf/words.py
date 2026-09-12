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
