"""tiktoken-backed tokenizer for OpenAI-family models."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=16)
def _encoding(model: str | None):
    import tiktoken

    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            pass
    # o200k_base covers gpt-4o / o1 families; cl100k_base is the safe general default.
    name = "o200k_base" if model and ("gpt-4o" in model or model.startswith("o1")) else "cl100k_base"
    return tiktoken.get_encoding(name)


class TiktokenTokenizer:
    def __init__(self, model: str | None = None):
        self._enc = _encoding(model)
        self.name = f"tiktoken:{getattr(self._enc, 'name', model or 'cl100k_base')}"

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def count(self, text: str) -> int:
        return len(self.encode(text))
