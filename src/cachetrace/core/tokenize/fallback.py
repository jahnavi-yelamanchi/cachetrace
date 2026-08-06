"""Pure-Python fallback tokenizer — no ML dependencies.

Not a real BPE, but a deterministic, GPT-ish word/subword splitter that produces
stable integer ids and a realistic token *count* (~0.75 words/token, long words
split into ~4-char pieces). Good enough to make the radix / prefix analysis
meaningful when neither ``tiktoken`` nor ``transformers`` is installed. The engine
warns that numbers are approximate in this mode.
"""

from __future__ import annotations

import re

# Split into words, whitespace runs, and individual punctuation — mirrors how BPE
# tends to break text at word boundaries.
_SPLIT_RE = re.compile(r"\s+|[A-Za-z0-9]+|[^\sA-Za-z0-9]")

_VOCAB = 100_000


def _stable_id(piece: str) -> int:
    # FNV-1a over the bytes → deterministic across processes (unlike hash()).
    h = 0x811C9DC5
    for b in piece.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % _VOCAB


def _subpieces(piece: str) -> list[str]:
    # Approximate BPE: long alnum runs get chopped into ~4-char subwords.
    if piece.isspace() or len(piece) <= 4 or not piece[0].isalnum():
        return [piece]
    return [piece[i : i + 4] for i in range(0, len(piece), 4)]


class FallbackTokenizer:
    name = "fallback"

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in _SPLIT_RE.findall(text):
            for sub in _subpieces(match):
                ids.append(_stable_id(sub))
        return ids

    def count(self, text: str) -> int:
        return len(self.encode(text))
