"""Per-family variation analysis: what varies, where, and why.

Each path is decomposed into constant + varying pieces; every varying piece is
classified independently so multiple busters in one system prompt (a timestamp
*and* a UUID) are reported separately. Shared by attribution (to blame tokens) and
the fix engine (to emit rules), so both agree on the exact set of busters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cachetrace.core.attribute.classify import REORDERING, Classification, classify_variation
from cachetrace.core.attribute.diff import Piece, Varying, decompose
from cachetrace.core.attribute.family import Family
from cachetrace.core.attribute.paths import path_value
from cachetrace.core.tokenize.render import render_segments

# Stable id for one varying span: (path, piece_index). path 'tools' => tool ordering.
SpanKey = tuple


@dataclass
class SpanVariation:
    path: str
    piece_index: int
    varying: Varying
    cls: Classification

    @property
    def key(self) -> SpanKey:
        return (self.path, self.piece_index)

    @property
    def n_busted(self) -> int:
        vals = self.varying.middles
        if not vals:
            return 0
        mode = max(set(vals), key=vals.count)
        return sum(1 for v in vals if v != mode)

    def examples(self, k: int = 3) -> list[str]:
        seen: list[str] = []
        for m in self.varying.middles:
            if m and m not in seen:
                seen.append(m)
            if len(seen) >= k:
                break
        return seen


@dataclass
class FamilyReport:
    family: Family
    spans: list[SpanVariation] = field(default_factory=list)
    decomp: dict[str, list[Piece]] = field(default_factory=dict)  # path -> decomposition

    @property
    def normalizable(self) -> list[SpanVariation]:
        return [s for s in self.spans if s.cls.normalizable]

    @property
    def variations(self) -> list[SpanVariation]:  # back-compat alias
        return self.spans


def analyze_family(family: Family) -> FamilyReport:
    reqs = family.requests
    ref = reqs[0]
    spans: list[SpanVariation] = []
    decomp: dict[str, list[Piece]] = {}

    for seg in render_segments(ref):
        if seg.path.startswith("tools["):
            continue  # tool positions handled via ordering check below
        values = [path_value(r, seg.path) for r in reqs]
        if all(v == values[0] for v in values):
            continue
        pieces = decompose(values)
        decomp[seg.path] = pieces
        for idx, piece in enumerate(pieces):
            if isinstance(piece, Varying):
                cls = classify_variation(piece.middles, left=piece.left, right=piece.right)
                spans.append(SpanVariation(path=seg.path, piece_index=idx, varying=piece, cls=cls))

    # Tool-list ordering (order-independent clustering routes reordered tools here).
    tool_seqs = ["".join(t.name for t in r.tools) for r in reqs]
    if len(reqs[0].tools) >= 2 and len(set(tool_seqs)) > 1:
        piece = Varying(middles=tool_seqs)
        cls = Classification(
            REORDERING, True, "tool list serialized in a different order each request", ""
        )
        spans.append(SpanVariation(path="tools", piece_index=0, varying=piece, cls=cls))

    return FamilyReport(family=family, spans=spans, decomp=decomp)
