"""Anthropic Messages API ingest.

The Messages API differs from OpenAI: the system prompt is a top-level ``system``
field (string or list of blocks), ``content`` is a list of typed blocks, and tools
carry ``input_schema`` instead of ``parameters``.
"""

from __future__ import annotations

from typing import Any

from cachetrace.core.model import Message, Request, ToolSpec

_PARAM_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "stop_sequences", "stream", "tool_choice")


def _parse_tools(raw: Any) -> list[ToolSpec]:
    tools: list[ToolSpec] = []
    for t in raw or []:
        if not isinstance(t, dict):
            continue
        tools.append(
            ToolSpec(
                name=str(t.get("name", "")),
                description=str(t.get("description", "")),
                parameters=t.get("input_schema", t.get("parameters", {})) or {},
            )
        )
    return tools


def _system_messages(system: Any) -> list[Message]:
    if not system:
        return []
    if isinstance(system, str):
        return [Message(role="system", content=system)]
    if isinstance(system, list):
        text = "".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in system)
        return [Message(role="system", content=text)]
    return []


def from_dict(obj: dict[str, Any], *, request_id: str, provider: str = "anthropic") -> Request:
    messages = _system_messages(obj.get("system"))
    for m in obj.get("messages", []) or []:
        if isinstance(m, dict):
            messages.append(Message(role=str(m.get("role", "user")), content=m.get("content")))
    params = {k: obj[k] for k in _PARAM_KEYS if k in obj}
    return Request(
        id=str(obj.get("id") or request_id),
        model=obj.get("model"),
        provider=provider,
        messages=messages,
        tools=_parse_tools(obj.get("tools")),
        params=params,
        arrival_ts=obj.get("arrival_ts") or obj.get("timestamp"),
        metadata=obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {},
    )
