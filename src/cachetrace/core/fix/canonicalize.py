"""Apply canonicalization to requests based on a family's variation report.

For each request we rebuild every varying segment from its decomposition:
constants are kept, normalizable volatile spans are collapsed to a family-constant
placeholder, and genuine per-request content is preserved as-is. Reordered tool
lists are sorted. After canonicalization every request in the family shares the
maximal prefix.

Used both to *measure* the achievable hit rate and, via the rules it implies, to
*emit* the fix.
"""

from __future__ import annotations

from cachetrace.core.attribute.diff import Constant, Varying
from cachetrace.core.attribute.paths import set_path_value
from cachetrace.core.attribute.report import FamilyReport, SpanKey
from cachetrace.core.model import Request


def _family_index(report: FamilyReport, req: Request) -> int | None:
    for i, r in enumerate(report.family.requests):
        if r is req:
            return i
    return None


def canonicalize_request(
    req: Request,
    report: FamilyReport,
    apply: set[SpanKey] | None = None,
) -> Request:
    """Return a canonicalized copy of ``req``.

    ``apply`` optionally restricts which spans (by ``(path, piece_index)``) are
    canonicalized — used for ordered ablation. ``None`` applies every normalizable
    span.
    """
    normalizable_keys = {s.key for s in report.normalizable}
    active = normalizable_keys if apply is None else (normalizable_keys & apply)

    out = req
    k = _family_index(report, req)

    # Rebuild each varying content path from its decomposition.
    for path, pieces in report.decomp.items():
        if not any((path, idx) in active for idx, p in enumerate(pieces) if isinstance(p, Varying)):
            continue
        parts: list[str] = []
        for idx, piece in enumerate(pieces):
            if isinstance(piece, Constant):
                parts.append(piece.text)
            else:  # Varying
                span = next((s for s in report.spans if s.key == (path, idx)), None)
                if (path, idx) in active and span is not None:
                    parts.append(span.cls.canonical)
                elif k is not None and k < len(piece.middles):
                    parts.append(piece.middles[k])  # keep real per-request content
                else:
                    parts.append(piece.middles[0] if piece.middles else "")
        out = set_path_value(out, path, "".join(parts))

    # Tool ordering.
    if ("tools", 0) in active:
        out = out.model_copy(deep=True)
        out.tools = sorted(out.tools, key=lambda t: t.name)

    return out
