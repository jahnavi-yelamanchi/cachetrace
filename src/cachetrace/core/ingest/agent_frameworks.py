"""Generic agent-framework trace ingest (LangChain / LlamaIndex / custom loggers).

These loggers wrap the real request in different envelopes. We look for the LLM
payload wherever it commonly hides (``input``, ``kwargs``, ``request``, ``body``,
``serialized``) and normalize a chat-message list or a single ``prompt`` string.
"""

from __future__ import annotations

from typing import Any

from cachetrace.core.ingest import openai
from cachetrace.core.model import Message, Request

_ENVELOPES = ("input", "kwargs", "request", "body", "data", "payload")


def _unwrap(obj: dict[str, Any]) -> dict[str, Any]:
    seen = 0
    while seen < 4 and isinstance(obj, dict) and "messages" not in obj and "prompt" not in obj:
        for key in _ENVELOPES:
            inner = obj.get(key)
            if isinstance(inner, dict):
                obj = inner
                break
        else:
            break
        seen += 1
    return obj


def from_dict(obj: dict[str, Any], *, request_id: str, provider: str = "agent") -> Request:
    inner = _unwrap(obj)

    if "messages" in inner:
        req = openai.from_dict(inner, request_id=request_id, provider=provider)
        req.model = req.model or obj.get("model")
        return req

    # Single flat prompt string (LangChain LLM traces, ReAct scratchpads, etc.)
    prompt = inner.get("prompt") or inner.get("text") or ""
    if isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)
    return Request(
        id=str(obj.get("id") or inner.get("id") or request_id),
        model=inner.get("model") or obj.get("model"),
        provider=provider,
        messages=[Message(role="user", content=str(prompt))],
        arrival_ts=obj.get("arrival_ts") or obj.get("timestamp"),
    )
