"""Render a canonical Request into the token stream an engine would prefill.

Two responsibilities:

* :func:`render_segments` lays the request out in engine order — system prompt,
  then tools, then message turns — as an ordered list of :class:`RenderSegment`,
  each tagged with a template path (``system``, ``tools[2]``, ``messages[4]``).
  It renders *faithfully*: it never reorders or canonicalizes, so genuine
  cache-busters (e.g. a per-request tool ordering) survive into the analysis.

* :func:`tokenize_request` encodes each segment with the chosen tokenizer and
  concatenates the ids, recording the token range each segment occupies. That
  range map is what attribution uses to name the field that busted the cache.
"""

from __future__ import annotations

import json

from cachetrace.core.model import RenderSegment, Request, TokenStream, ToolSpec
from cachetrace.core.tokenize.base import Tokenizer


def _tool_text(tool: ToolSpec) -> str:
    # Preserve field order as given (do NOT sort) so reordering busters are visible.
    payload = {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_segments(req: Request) -> list[RenderSegment]:
    segments: list[RenderSegment] = []

    # 1) System prompt(s) — the most cache-sensitive region.
    for i, m in enumerate(req.messages):
        if m.role == "system":
            path = "system" if i == 0 else f"system[{i}]"
            segments.append(RenderSegment(path=path, text=f"<|system|>\n{m.text()}\n"))

    # 2) Tools block — list order is significant and a common buster.
    for i, tool in enumerate(req.tools):
        segments.append(RenderSegment(path=f"tools[{i}]", text=f"<|tool|>{_tool_text(tool)}\n"))

    # 3) Conversation turns in order.
    for i, m in enumerate(req.messages):
        if m.role == "system":
            continue
        volatile = m.role in ("tool", "function")
        segments.append(
            RenderSegment(
                path=f"messages[{i}]",
                text=f"<|{m.role}|>\n{m.text()}\n",
                volatile_hint=volatile,
            )
        )
    return segments


def tokenize_request(req: Request, tokenizer: Tokenizer) -> TokenStream:
    token_ids: list[int] = []
    spans: list[tuple[str, int, int]] = []
    for seg in render_segments(req):
        start = len(token_ids)
        token_ids.extend(tokenizer.encode(seg.text))
        spans.append((seg.path, start, len(token_ids)))
    return TokenStream(request_id=req.id, token_ids=token_ids, segments=spans)
