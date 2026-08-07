"""AWS Bedrock Converse API ingest.

Converse uses ``messages:[{role, content:[{text}|{toolUse}|{toolResult}]}]``, a
top-level ``system:[{text}]`` list, and tools under ``toolConfig.tools[].toolSpec``
with an ``inputSchema.json`` schema.
"""

from __future__ import annotations

from typing import Any

from cachetrace.core.model import Message, Request, ToolSpec


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if "text" in b:
            chunks.append(str(b["text"]))
        elif "toolUse" in b:
            tu = b["toolUse"]
            chunks.append(f"{tu.get('name', '')}({tu.get('input', '')})")
        elif "toolResult" in b:
            chunks.append(str(b["toolResult"].get("content", "")))
    return "".join(chunks)


def _system(obj: dict[str, Any]) -> list[Message]:
    sys = obj.get("system")
    if not sys:
        return []
    if isinstance(sys, str):
        return [Message(role="system", content=sys)]
    return [Message(role="system", content=_blocks_text(sys))]


def _tools(obj: dict[str, Any]) -> list[ToolSpec]:
    tools: list[ToolSpec] = []
    cfg = obj.get("toolConfig", {}) or {}
    for t in cfg.get("tools", []) or []:
        spec = t.get("toolSpec", t) if isinstance(t, dict) else {}
        schema = spec.get("inputSchema", {}) or {}
        params = schema.get("json", schema) if isinstance(schema, dict) else {}
        tools.append(
            ToolSpec(
                name=str(spec.get("name", "")),
                description=str(spec.get("description", "")),
                parameters=params or {},
            )
        )
    return tools


def from_dict(obj: dict[str, Any], *, request_id: str, provider: str = "bedrock") -> Request:
    messages = _system(obj)
    for m in obj.get("messages", []) or []:
        if isinstance(m, dict):
            messages.append(Message(role=str(m.get("role", "user")), content=_blocks_text(m.get("content"))))
    params = obj.get("inferenceConfig", {}) if isinstance(obj.get("inferenceConfig"), dict) else {}
    return Request(
        id=str(obj.get("id") or request_id),
        model=obj.get("modelId") or obj.get("model"),
        provider=provider,
        messages=messages,
        tools=_tools(obj),
        params=params,
        arrival_ts=obj.get("arrival_ts") or obj.get("timestamp"),
    )
