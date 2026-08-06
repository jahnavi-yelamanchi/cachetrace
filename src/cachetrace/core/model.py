"""Canonical data model for cachetrace.

Every provider format (OpenAI, Anthropic, Gemini, Bedrock, agent-framework traces, …)
is normalized into a :class:`Request`. Analysis never touches the raw provider payload
again — it works on ``Request`` objects and the :class:`TokenStream` they render into.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """A tool/function definition available to the model for a request."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single chat message in canonical form.

    ``content`` may be a plain string or a list of content parts (as used by
    multimodal / Anthropic-style messages). We keep it flexible and stringify at
    render time.
    """

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def text(self) -> str:
        """Best-effort flat text of the message content (for rendering/diffing)."""
        if self.content is None:
            parts: list[str] = []
            for call in self.tool_calls or []:
                fn = call.get("function", call)
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                parts.append(f"{name}({args})")
            return "\n".join(parts)
        if isinstance(self.content, str):
            return self.content
        # list of parts
        chunks: list[str] = []
        for part in self.content:
            if isinstance(part, dict):
                chunks.append(str(part.get("text", part.get("content", ""))))
            else:
                chunks.append(str(part))
        return "".join(chunks)


class Request(BaseModel):
    """A single canonical inference request."""

    id: str
    model: str | None = None
    provider: str | None = None
    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    arrival_ts: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def system_text(self) -> str:
        return "\n".join(m.text() for m in self.messages if m.role == "system")


class RenderSegment(BaseModel):
    """One logical piece of a rendered prompt, tagged with a template path.

    The renderer emits an ordered list of these. Concatenating ``text`` reconstructs
    the full prompt string; the ``path`` (e.g. ``"system"``, ``"tools[0].name"``,
    ``"messages[2].content"``) is what attribution reports when a segment busts the
    cache.
    """

    path: str
    text: str
    # True when the content is inherently per-request volatile (e.g. a tool result);
    # the renderer/attribution use this as a hint only.
    volatile_hint: bool = False


class TokenStream(BaseModel):
    """A rendered request as an engine would see it: a flat token sequence plus a
    map from token ranges back to template paths.
    """

    request_id: str
    token_ids: list[int] = Field(default_factory=list)
    # (path, start_token_index, end_token_index_exclusive)
    segments: list[tuple[str, int, int]] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.token_ids)

    def path_at(self, token_index: int) -> str | None:
        """Return the template path that produced the token at ``token_index``."""
        for path, start, end in self.segments:
            if start <= token_index < end:
                return path
        return None


class RequestBatch(BaseModel):
    """An ordered collection of requests (arrival order preserved)."""

    requests: list[Request] = Field(default_factory=list)

    def __iter__(self) -> Iterator[Request]:  # type: ignore[override]
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)

    @classmethod
    def from_iterable(cls, items: Iterable[Request]) -> RequestBatch:
        return cls(requests=list(items))

    def to_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        with p.open("w", encoding="utf-8") as fh:
            for req in self.requests:
                fh.write(req.model_dump_json(exclude_none=True) + "\n")
