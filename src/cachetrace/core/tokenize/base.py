"""Tokenizer protocol shared by every backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal interface the analysis engine needs from a tokenizer.

    Backends must be deterministic: identical input text must always encode to the
    identical token id sequence, or prefix matching breaks.
    """

    name: str

    def encode(self, text: str) -> list[int]:
        ...

    def count(self, text: str) -> int:
        ...
