"""Ingest layer: normalize any provider's request trace into canonical Requests.

Public entry point is :func:`load_requests`, which streams a JSONL file and
auto-detects the provider format per line (unless one is forced).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cachetrace.core.ingest import (
    agent_frameworks,
    anthropic,
    bedrock,
    gemini,
    openai,
    vllm_log,
)
from cachetrace.core.model import Request, RequestBatch

__all__ = ["load_requests", "load_batch", "detect_provider", "from_dict"]

_ADAPTERS = {
    "anthropic": anthropic.from_dict,
    "gemini": gemini.from_dict,
    "bedrock": bedrock.from_dict,
    "agent": agent_frameworks.from_dict,
    "vllm": vllm_log.from_dict,
    "openai": openai.from_dict,
}


def _is_block_list(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and ("text" in b or "toolUse" in b or "toolResult" in b) for b in content
    )


def detect_provider(obj: dict[str, Any]) -> str:
    """Guess which provider format a raw request dict is in."""
    if "contents" in obj or "systemInstruction" in obj or "system_instruction" in obj:
        return "gemini"
    if "toolConfig" in obj or "modelId" in obj:
        return "bedrock"
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs and _is_block_list(msgs[0].get("content") if isinstance(msgs[0], dict) else None):
        return "bedrock"
    if "system" in obj and "messages" in obj:
        return "anthropic"  # Anthropic keeps system out of the messages array
    if any(isinstance(t, dict) and "input_schema" in t for t in obj.get("tools", []) or []):
        return "anthropic"
    # Agent-framework envelopes hide the payload under a wrapper key.
    if "messages" not in obj and "prompt" not in obj and any(
        isinstance(obj.get(k), dict) for k in ("input", "kwargs", "request", "body")
    ):
        return "agent"
    if "prompt" in obj and "messages" not in obj:
        return "agent"
    if "messages" in obj:
        return "openai"
    return "openai"


def from_dict(obj: dict[str, Any], *, request_id: str, provider: str | None = None) -> Request:
    provider = provider or detect_provider(obj)
    adapter = _ADAPTERS.get(provider, openai.from_dict)
    return adapter(obj, request_id=request_id, provider=provider)


def load_requests(
    path: str | Path,
    *,
    provider: str | None = None,
    limit: int | None = None,
) -> Iterator[Request]:
    """Stream canonical Requests from a JSONL trace file.

    Each line may be a raw provider request or a previously-exported canonical
    Request. Blank lines and malformed lines are skipped. Memory usage stays flat
    regardless of file size.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        count = 0
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            req = from_dict(obj, request_id=f"req-{lineno}", provider=provider)
            yield req
            count += 1
            if limit is not None and count >= limit:
                return


def load_batch(
    path: str | Path,
    *,
    provider: str | None = None,
    limit: int | None = None,
) -> RequestBatch:
    return RequestBatch.from_iterable(load_requests(path, provider=provider, limit=limit))
