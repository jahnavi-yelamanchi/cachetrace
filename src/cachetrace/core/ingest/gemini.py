"""Google Gemini / Vertex `generateContent` ingest.

Gemini uses ``contents:[{role, parts:[{text}]}]`` with roles ``user``/``model``,
a top-level ``systemInstruction``, and tools under
``tools[].functionDeclarations``.
"""

from __future__ import annotations

from typing import Any

from cachetrace.core.model import Message, Request, ToolSpec

_ROLE_MAP = {"model": "assistant", "user": "user"}


def _parts_text(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    chunks: list[str] = []
    for p in parts or []:
        if isinstance(p, dict):
            if "text" in p:
                chunks.append(str(p["text"]))
            elif "functionCall" in p:
                fc = p["functionCall"]
                chunks.append(f"{fc.get('name', '')}({fc.get('args', '')})")
    return "".join(chunks)


def _system(obj: dict[str, Any]) -> list[Message]:
    si = obj.get("systemInstruction") or obj.get("system_instruction")
    if not si:
        return []
    if isinstance(si, str):
        return [Message(role="system", content=si)]
    return [Message(role="system", content=_parts_text(si.get("parts", si)))]


def _tools(obj: dict[str, Any]) -> list[ToolSpec]:
    tools: list[ToolSpec] = []
    for t in obj.get("tools", []) or []:
        decls = t.get("functionDeclarations", t.get("function_declarations", [])) if isinstance(t, dict) else []
        for fn in decls or []:
            tools.append(
                ToolSpec(
                    name=str(fn.get("name", "")),
                    description=str(fn.get("description", "")),
                    parameters=fn.get("parameters", {}) or {},
                )
            )
    return tools


def from_dict(obj: dict[str, Any], *, request_id: str, provider: str = "gemini") -> Request:
    messages = _system(obj)
    for c in obj.get("contents", []) or []:
        if isinstance(c, dict):
            role = _ROLE_MAP.get(c.get("role", "user"), "user")
            messages.append(Message(role=role, content=_parts_text(c.get("parts"))))
    params = {}
    if isinstance(obj.get("generationConfig"), dict):
        params = obj["generationConfig"]
    return Request(
        id=str(obj.get("id") or request_id),
        model=obj.get("model"),
        provider=provider,
        messages=messages,
        tools=_tools(obj),
        params=params,
        arrival_ts=obj.get("arrival_ts") or obj.get("timestamp"),
    )
