"""Selection and pre-value boundaries for generated arithmetic chains."""
from __future__ import annotations

import hashlib
import re


def fixed_sample(set_ids, n=200):
    return set(sorted(set(set_ids), key=lambda value:
        hashlib.sha256(f"round5:{value}".encode()).hexdigest())[:n])


def value_boundary(generation, name):
    """First assignment clause ending in an integer; expression operands are not value writes."""
    head = generation.split("\n\n", 1)[0].split("Answer:", 1)[0]
    for match in re.finditer(rf"\b{re.escape(name)}\s*=\s*(?P<body>[^,\n;]*)", head):
        body = match.group("body")
        value = re.search(r"(?:^|=)\s*(?P<value>-?\d+)\s*$", body)
        if value is None:
            continue
        start = match.start("body") + value.start("value")
        marker = head.rfind("=", match.start(), start)
        return {"marker": marker, "value_start": start, "value": int(value.group("value"))}
    return None
