"""Cluster requests into template families.

Requests only share a cache prefix if they were built from the same template. We
group by a structural signature — the sequence of message roles, the tool names,
and the set of rendered segment paths — so distinct prompts are never wrongly
compared. Within a family we then look for what varies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cachetrace.core.model import Request


def signature(req: Request) -> tuple:
    roles = tuple(m.role for m in req.messages)
    # Order-independent so reordered tool lists cluster together (and the ordering
    # itself is then detectable as a buster rather than splitting the family).
    tools = frozenset(t.name for t in req.tools)
    n_tools = len(req.tools)
    return (roles, tools, n_tools)


@dataclass
class Family:
    key: tuple
    requests: list[Request] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.requests)


def cluster_families(requests: list[Request]) -> list[Family]:
    families: dict[tuple, Family] = {}
    for req in requests:
        sig = signature(req)
        fam = families.get(sig)
        if fam is None:
            fam = families[sig] = Family(key=sig)
        fam.requests.append(req)
    # Largest families first — they dominate the waste.
    return sorted(families.values(), key=lambda f: f.size, reverse=True)
