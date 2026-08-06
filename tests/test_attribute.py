"""Diff decomposition, root-cause classification, and waste attribution."""

from __future__ import annotations

from cachetrace.core.attribute import run_attribution
from cachetrace.core.attribute.classify import (
    COUNTER,
    REORDERING,
    TIMESTAMP,
    UUID,
    classify_variation,
)
from cachetrace.core.attribute.diff import varying_pieces
from cachetrace.core.tokenize.fallback import FallbackTokenizer


def test_decompose_separates_two_volatile_spans():
    values = [
        "time: 2026-08-06T10:01:00Z. session: 111e1111-1111-1111-1111-111111111111. go",
        "time: 2026-08-06T10:02:00Z. session: 222e2222-2222-2222-2222-222222222222. go",
        "time: 2026-08-06T10:03:00Z. session: 333e3333-3333-3333-3333-333333333333. go",
    ]
    varyings = varying_pieces(values)
    # two independent varying spans (timestamp, uuid), not one merged blob
    assert len(varyings) == 2


def test_decompose_constant_has_no_varying():
    assert varying_pieces(["same", "same", "same"]) == []


def test_classify_timestamp_with_context():
    # middle stripped of shared prefix; context makes it recognizable as ISO time
    c = classify_variation(["06T10:01:00", "06T10:02:00"], left="2026-08-", right="Z")
    assert c.cause == TIMESTAMP
    assert c.normalizable


def test_classify_uuid():
    c = classify_variation(
        ["111e1111-1111-1111-1111-111111111111", "222e2222-2222-2222-2222-222222222222"]
    )
    assert c.cause == UUID


def test_classify_counter():
    c = classify_variation(["1", "2", "3", "4"])
    assert c.cause == COUNTER


def test_classify_reordering():
    c = classify_variation(["alpha, beta, gamma", "gamma, alpha, beta", "beta, gamma, alpha"])
    assert c.cause == REORDERING


def test_classify_real_content_is_not_normalizable():
    c = classify_variation(["Where is my order?", "I want a refund.", "Ship it faster"])
    assert not c.normalizable


def test_attribution_blames_the_right_causes(messy_trace):
    tok = FallbackTokenizer()
    result = run_attribution(messy_trace, tok, block_size=16)
    causes = {b.cause for b in result.busters}
    assert TIMESTAMP in causes
    assert UUID in causes
    assert REORDERING in causes
    assert result.total_recoverable_tokens > 0
    # every buster points at a concrete template location
    assert all(b.path for b in result.busters)


def test_clean_trace_has_no_normalizable_busters(clean_trace):
    tok = FallbackTokenizer()
    result = run_attribution(clean_trace, tok, block_size=16)
    assert result.busters == []
