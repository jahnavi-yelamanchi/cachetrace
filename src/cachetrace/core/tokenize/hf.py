"""HuggingFace-backed tokenizer for open models (Llama, Qwen, Mistral, …).

Uses the model's real BPE/SentencePiece vocabulary. The chat-template *structure*
is handled by our segment-aware renderer (see ``render.py``) so we keep the
token→template-path mapping that attribution needs; this class only encodes text.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=8)
def _load(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


class HFTokenizer:
    def __init__(self, model: str):
        self._tok = _load(model)
        self.name = f"hf:{model}"

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def count(self, text: str) -> int:
        return len(self.encode(text))
