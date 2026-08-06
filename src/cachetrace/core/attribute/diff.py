"""Decompose a family's differing strings into constant and varying pieces.

A single system prompt often carries several independent busters — a timestamp
*and* a session UUID. Stripping one outer prefix/suffix would merge them into one
blob. Instead we recursively split on the longest constant substring shared by all
values, yielding an ordered decomposition:

    Constant("... Current time: ") Varying(<timestamps>) Constant(". Session: ")
    Varying(<uuids>) Constant(". Answer ...")

Each ``Varying`` piece is classified on its own, so the timestamp and the UUID are
reported (and priced) separately. Concatenating the pieces reconstructs every
value, which is also how canonicalization rebuilds the segment.
"""

from __future__ import annotations

from dataclasses import dataclass

# Minimum length of a shared substring we'll treat as a real constant anchor.
# Large enough that a timestamp's internal ":" / "-" separators don't split it,
# small enough to catch phrase separators like ". Session: ".
_MIN_ANCHOR = 5


@dataclass
class Constant:
    text: str


@dataclass
class Varying:
    middles: list[str]  # per-request value of this span (family order)
    left: str = ""      # adjacent constant context (for classification/regex)
    right: str = ""


Piece = object  # Union[Constant, Varying]


def _common_prefix(values: list[str]) -> str:
    s1, s2 = min(values), max(values)
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]


def _longest_common_substring(values: list[str]) -> str:
    """Longest substring present in every value (anchored on the shortest one)."""
    if not values:
        return ""
    base = min(values, key=len)
    n = len(base)
    best = ""
    for i in range(n):
        # Only need to beat the current best length.
        for j in range(i + len(best) + 1, n + 1):
            cand = base[i:j]
            if all(cand in v for v in values):
                if len(cand) > len(best):
                    best = cand
            else:
                break
    return best


def _decompose(values: list[str], left: str, right: str) -> list[Piece]:
    if all(v == values[0] for v in values):
        return [Constant(values[0])] if values[0] else []

    prefix = _common_prefix(values)
    rem = [v[len(prefix):] for v in values]
    rev_suffix = _common_prefix([r[::-1] for r in rem])
    suffix = rev_suffix[::-1]
    slen = len(suffix)
    middles = [r[: len(r) - slen] if slen else r for r in rem]

    pieces: list[Piece] = []
    if prefix:
        pieces.append(Constant(prefix))

    if all(m == middles[0] for m in middles):
        if middles[0]:
            pieces.append(Constant(middles[0]))
    else:
        anchor = _longest_common_substring(middles)
        if len(anchor) >= _MIN_ANCHOR and all(anchor in m for m in middles):
            lefts = [m.split(anchor, 1)[0] for m in middles]
            rights = [m.split(anchor, 1)[1] for m in middles]
            pieces.extend(_decompose(lefts, left + prefix, anchor))
            pieces.append(Constant(anchor))
            pieces.extend(_decompose(rights, anchor, suffix + right))
        else:
            pieces.append(Varying(middles=middles, left=left + prefix, right=suffix + right))

    if suffix:
        pieces.append(Constant(suffix))
    return _merge_constants(pieces)


def _merge_constants(pieces: list[Piece]) -> list[Piece]:
    out: list[Piece] = []
    for p in pieces:
        if isinstance(p, Constant) and out and isinstance(out[-1], Constant):
            out[-1] = Constant(out[-1].text + p.text)
        else:
            out.append(p)
    return out


def decompose(values: list[str]) -> list[Piece]:
    """Ordered Constant/Varying decomposition of a family's segment values."""
    if not values:
        return []
    return _decompose(values, "", "")


def varying_pieces(values: list[str]) -> list[Varying]:
    return [p for p in decompose(values) if isinstance(p, Varying)]
