"""Read/write the payload text of a rendered segment path on a Request.

Attribution diffs these payloads; the fix engine rewrites them. Keeping both on
one accessor guarantees they agree on what a path like ``system`` or
``messages[3]`` refers to.
"""

from __future__ import annotations

import re

from cachetrace.core.model import Message, Request
from cachetrace.core.tokenize.render import _tool_text

_IDX = re.compile(r"\[(\d+)\]")


def _index(path: str) -> int:
    m = _IDX.search(path)
    return int(m.group(1)) if m else 0


def _system_message_indices(req: Request) -> list[int]:
    return [i for i, m in enumerate(req.messages) if m.role == "system"]


def path_value(req: Request, path: str) -> str:
    """Return the payload text a path contributes to the rendered prompt."""
    if path.startswith("system"):
        idx = _index(path) if "[" in path else (_system_message_indices(req) or [0])[0]
        if 0 <= idx < len(req.messages):
            return req.messages[idx].text()
        return ""
    if path.startswith("tools["):
        i = _index(path)
        return _tool_text(req.tools[i]) if 0 <= i < len(req.tools) else ""
    if path.startswith("messages["):
        i = _index(path)
        return req.messages[i].text() if 0 <= i < len(req.messages) else ""
    return ""


def set_path_value(req: Request, path: str, new_text: str) -> Request:
    """Return a copy of ``req`` with ``path``'s payload replaced by ``new_text``."""
    copy = req.model_copy(deep=True)
    if path.startswith("system") or path.startswith("messages["):
        idx = _index(path) if "[" in path else (_system_message_indices(copy) or [0])[0]
        if 0 <= idx < len(copy.messages):
            copy.messages[idx] = Message(
                role=copy.messages[idx].role,
                content=new_text,
                name=copy.messages[idx].name,
                tool_call_id=copy.messages[idx].tool_call_id,
            )
    return copy
