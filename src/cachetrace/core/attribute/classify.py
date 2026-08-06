"""Classify the root cause of a varying span across requests.

Given the differing middles of a segment (from :mod:`diff`), decide *why* they
differ and whether the difference is safely normalizable (noise the fix can hoist
or canonicalize) versus legitimate per-request content that must be preserved.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Root-cause labels.
TIMESTAMP = "timestamp"
UUID = "uuid"
RANDOM_ID = "random_id"
COUNTER = "counter"
REORDERING = "reordering"
NONDETERMINISTIC_JSON = "nondeterministic_json"
VARIABLE_CONTENT = "variable_content"

_ISO = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s?(AM|PM|am|pm)?\b")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_HEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_LONG_TOKEN = re.compile(r"\b[A-Za-z0-9_-]{16,}\b")
_INT = re.compile(r"-?\d+")

_HUMAN_HINT = {
    TIMESTAMP: "a timestamp in the cached prefix — every request differs, so nothing after it caches",
    UUID: "a per-request UUID busting the prefix",
    RANDOM_ID: "a high-entropy per-request id (session/trace/request id) busting the prefix",
    COUNTER: "a monotonically increasing counter busting the prefix",
    REORDERING: "the same fields serialized in a different order each request",
    NONDETERMINISTIC_JSON: "JSON keys emitted in non-deterministic order",
    VARIABLE_CONTENT: "genuine per-request content (not necessarily fixable)",
}


@dataclass
class Classification:
    cause: str
    normalizable: bool  # can the fix safely canonicalize/hoist this?
    detail: str
    canonical: str  # what the span becomes after canonicalization


def _entropy(values: list[str]) -> float:
    joined = "".join(values)
    if not joined:
        return 0.0
    freq: dict[str, int] = {}
    for ch in joined:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(joined)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _all_match(values: list[str], pattern: re.Pattern) -> bool:
    return all(pattern.search(v) for v in values) and any(v.strip() for v in values)


def _tokenize_items(s: str) -> list[str]:
    # split into comma / newline / semicolon separated items for reorder detection
    parts = re.split(r"[,\n;]", s)
    return [p.strip() for p in parts if p.strip()]


def _is_reordering(values: list[str]) -> bool:
    sets = [frozenset(_tokenize_items(v)) for v in values]
    seqs = [tuple(_tokenize_items(v)) for v in values]
    if any(len(s) < 2 for s in sets):
        return False
    # same multiset of items, but not all in the same order
    return len(set(sets)) == 1 and len(set(seqs)) > 1


def classify_variation(values: list[str], left: str = "", right: str = "") -> Classification:
    """Classify the differing spans across a family of requests.

    ``left``/``right`` are the constant text adjacent to the span; the regex
    detectors run against ``left + value + right`` so an entity that shares a
    constant fragment with its neighbours (``2026-08-`` before every timestamp) is
    still recognized. Reorder/counter checks run on the raw spans.
    """
    clean = [v.strip() for v in values]
    ctx = [left + v + right for v in values]

    if _is_reordering(values):
        items = sorted(_tokenize_items(values[0]))
        return Classification(REORDERING, True, _HUMAN_HINT[REORDERING], ", ".join(items))

    if _all_match(ctx, _UUID):
        return Classification(UUID, True, _HUMAN_HINT[UUID], "<uuid>")

    if _all_match(ctx, _ISO) or _all_match(ctx, _DATE) or _all_match(ctx, _TIME):
        return Classification(TIMESTAMP, True, _HUMAN_HINT[TIMESTAMP], "<timestamp>")

    # Pure integers that increase → counter; otherwise varying integers still noise.
    if all(_INT.fullmatch(v) for v in clean if v) and any(clean):
        nums = [int(v) for v in clean if v]
        if nums == sorted(nums) and len(set(nums)) > 1:
            return Classification(COUNTER, True, _HUMAN_HINT[COUNTER], "<counter>")

    if _all_match(ctx, _HEX) or (_all_match(clean, _LONG_TOKEN) and _entropy(clean) > 3.0):
        return Classification(RANDOM_ID, True, _HUMAN_HINT[RANDOM_ID], "<id>")

    return Classification(VARIABLE_CONTENT, False, _HUMAN_HINT[VARIABLE_CONTENT], "")
