"""Apply an emitted `cachetrace_fix.yaml` ruleset to live request payloads.

Same semantics as the emitted Python transform, but rule-driven so the proxy can
"fix in production" from the YAML the offline `fix` command produced.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def load_rules(path: str | Path) -> list[dict[str, Any]]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return doc.get("cachetrace_fix", {}).get("rules", [])


def apply_rules(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    rules: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    messages = [dict(m) for m in messages]
    hoisted: list[str] = []

    for rule in rules:
        if rule.get("action") == "sort" and rule.get("target") == "tools" and tools:
            tools = sorted(tools, key=lambda t: (t.get("function", t) or {}).get("name", ""))
            continue
        pattern = rule.get("pattern")
        if not pattern:
            continue
        rx = re.compile(pattern, re.DOTALL)
        for m in messages:
            if rule.get("role") and m.get("role") != rule["role"]:
                continue
            content = m.get("content")
            if not isinstance(content, str):
                continue
            if rule.get("action") == "hoist" and "vol" in rx.groupindex:
                for match in rx.finditer(content):
                    hoisted.append(match.group("vol"))
            m["content"] = rx.sub(rule.get("replacement", ""), content)

    if hoisted:
        tail = " [context: " + " ".join(hoisted) + "]"
        if messages and isinstance(messages[-1].get("content"), str):
            messages[-1]["content"] += tail
        else:
            messages.append({"role": "user", "content": tail.strip()})
    return messages, tools
