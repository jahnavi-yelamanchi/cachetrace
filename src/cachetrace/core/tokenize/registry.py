"""Resolve a model id to a concrete tokenizer backend.

Selection order for ``backend='auto'``:

1. OpenAI-family model id + ``tiktoken`` installed  → tiktoken
2. HF repo id (``org/model``) + ``transformers`` installed → HF
3. otherwise → the dependency-free fallback tokenizer (with a warning)
"""

from __future__ import annotations

import importlib.util
import warnings
from functools import lru_cache

from cachetrace.core.tokenize.base import Tokenizer
from cachetrace.core.tokenize.fallback import FallbackTokenizer

_OPENAI_HINTS = ("gpt-", "gpt3", "gpt4", "o1", "o3", "o4", "text-embedding", "davinci", "chatgpt")


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _looks_openai(model: str | None) -> bool:
    if not model:
        return False
    m = model.lower()
    return any(h in m for h in _OPENAI_HINTS) and "/" not in m


def _looks_hf(model: str | None) -> bool:
    return bool(model) and "/" in model


def _tiktoken(model: str | None) -> Tokenizer:
    from cachetrace.core.tokenize.tiktoken_ import TiktokenTokenizer

    tok = TiktokenTokenizer(model)
    tok.encode("warmup")  # force vocab load now so failures surface here
    return tok


def _hf(model: str) -> Tokenizer:
    from cachetrace.core.tokenize.hf import HFTokenizer

    return HFTokenizer(model)


def _try(factory) -> Tokenizer | None:
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 — degrade gracefully on any load failure
        warnings.warn(f"tokenizer backend unavailable ({exc}); trying next.", stacklevel=3)
        return None


@lru_cache(maxsize=32)
def get_tokenizer(model: str | None = None, backend: str = "auto") -> Tokenizer:
    """Return a cached tokenizer for ``model`` using the requested ``backend``."""
    backend = (backend or "auto").lower()

    if backend == "fallback":
        return FallbackTokenizer()
    if backend == "tiktoken":
        return _tiktoken(model)
    if backend == "hf":
        if not model:
            raise ValueError("backend='hf' requires a model id (e.g. 'meta-llama/Llama-3.1-8B-Instruct')")
        return _hf(model)

    # auto — try real backends but never crash the audit if a vocab can't be
    # loaded (offline box, blocked egress, missing HF auth); degrade to fallback.
    if _looks_hf(model) and _has("transformers"):
        tok = _try(lambda: _hf(model))
        if tok is not None:
            return tok
    if _has("tiktoken"):
        tok = _try(lambda: _tiktoken(model))
        if tok is not None:
            return tok

    warnings.warn(
        "No tokenizer backend available (install cachetrace[tiktoken] or cachetrace[hf] "
        "for accurate token counts). Falling back to an approximate tokenizer.",
        stacklevel=2,
    )
    return FallbackTokenizer()


def describe_backend(model: str | None = None, backend: str = "auto") -> str:
    return get_tokenizer(model, backend).name
