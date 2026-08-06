"""OpenAI (and OpenAI-compatible) chat-completion ingest.

Handles the ``/v1/chat/completions`` request shape used by OpenAI, Azure OpenAI,
Together, Fireworks, vLLM/SGLang/TGI OpenAI servers, Groq, Mistral, DeepSeek, etc.
"""

from __future__ import annotations

from typing import Any

from cachetrace.core.model import Message, Request, ToolSpec


def _parse_tools(raw: Any) -> list[ToolSpec]:
    tools: list[ToolSpec] = []
    for t in raw or []:
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            fn = t["function"]
        else:
            fn = t
        if not isinstance(fn, dict):
            continue
        tools.append(
            ToolSpec(
                name=str(fn.get("name", "")),
                description=str(fn.get("description", "")),
                parameters=fn.get("parameters", {}) or {},
            )
        )
    return tools


def _parse_messages(raw: Any) -> list[Message]:
    messages: list[Message] = []
    for m in raw or []:
        if not isinstance(m, dict):
            continue
        messages.append(
            Message(
                role=str(m.get("role", "user")),
                content=m.get("content"),
                name=m.get("name"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
        )
    return messages


# Params that influence sampling/serving but not the prompt prefix.
_PARAM_KEYS = (
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop",
    "presence_penalty", "frequency_penalty", "seed", "n", "stream",
    "response_format", "tool_choice", "logprobs",
)


def from_dict(obj: dict[str, Any], *, request_id: str, provider: str = "openai") -> Request:
    """Normalize a single OpenAI chat-completion request dict into a Request."""
    params = {k: obj[k] for k in _PARAM_KEYS if k in obj}
    metadata = obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {}
    return Request(
        id=str(obj.get("id") or obj.get("request_id") or request_id),
        model=obj.get("model"),
        provider=provider,
        messages=_parse_messages(obj.get("messages")),
        tools=_parse_tools(obj.get("tools") or obj.get("functions")),
        params=params,
        arrival_ts=obj.get("arrival_ts") or obj.get("timestamp"),
        metadata=metadata,
    )
