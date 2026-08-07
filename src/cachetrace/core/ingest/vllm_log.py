"""vLLM / OpenAI-server request-log ingest.

Server access logs often wrap the chat-completion body under ``request``/``body``
alongside timing metadata. Unwrap and hand off to the OpenAI normalizer.
"""

from __future__ import annotations

from typing import Any

from cachetrace.core.ingest import openai
from cachetrace.core.model import Request


def from_dict(obj: dict[str, Any], *, request_id: str, provider: str = "vllm") -> Request:
    body = obj
    for key in ("request", "body", "payload"):
        inner = obj.get(key)
        if isinstance(inner, dict) and "messages" in inner:
            body = inner
            break
    req = openai.from_dict(body, request_id=request_id, provider=provider)
    if req.arrival_ts is None:
        req.arrival_ts = obj.get("timestamp") or obj.get("arrival_ts")
    return req
