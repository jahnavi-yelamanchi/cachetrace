"""Ingest layer: normalize any provider's request trace into canonical Requests.

Public entry point is :func:`load_requests`, which streams a JSONL file and
auto-detects the provider format per line (unless one is forced).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cachetrace.core.ingest import anthropic, openai
from cachetrace.core.model import Request, RequestBatch

__all__ = ["load_requests", "load_batch", "detect_provider", "from_dict"]


def detect_provider(obj: dict[str, Any]) -> str:
    """Guess which provider format a raw request dict is in."""
    if "system" in obj and "messages" in obj:
        # Anthropic keeps system out of the messages array.
        return "anthropic"
    if any(isinstance(t, dict) and "input_schema" in t for t in obj.get("tools", []) or []):
        return "anthropic"
    if "contents" in obj:  # Google Gemini
        return "gemini"
    if "messages" in obj:
        return "openai"
    return "openai"


def from_dict(obj: dict[str, Any], *, request_id: str, provider: str | None = None) -> Request:
    provider = provider or detect_provider(obj)
    if provider == "anthropic":
        return anthropic.from_dict(obj, request_id=request_id, provider=provider)
    # gemini/bedrock/etc. fall through to the OpenAI-shaped normalizer for now;
    # dedicated adapters are added in the providers-breadth milestone.
    return openai.from_dict(obj, request_id=request_id, provider=provider)


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
