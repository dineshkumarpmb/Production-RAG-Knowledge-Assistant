"""Approximate token counting for chunk sizing.

Exact counts depend on each embedding model's tokenizer, and chunk sizes only need to be
roughly right, so this counts words and punctuation and splits long words into ~6-character
pieces (English averages about 1.3 tokens per word with modern sub-word tokenizers).
"""

import math
import re

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return sum(math.ceil(len(tok) / 6) for tok in _TOKEN.findall(text))
