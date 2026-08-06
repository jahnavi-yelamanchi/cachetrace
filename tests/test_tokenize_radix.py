"""Tokenizer determinism + radix tree / simulator behavior."""

from __future__ import annotations

from cachetrace.core.radix.simulator import Simulator
from cachetrace.core.radix.tree import RadixTree, block_hashes
from cachetrace.core.tokenize.fallback import FallbackTokenizer
from cachetrace.core.tokenize.render import tokenize_request


def test_fallback_tokenizer_is_deterministic():
    tok = FallbackTokenizer()
    a = tok.encode("the quick brown fox jumps over the lazy dog")
    b = tok.encode("the quick brown fox jumps over the lazy dog")
    assert a == b
    # identical substrings encode identically (required for prefix matching)
    assert tok.encode("hello world")[:1] == tok.encode("hello there")[:1]


def test_block_hashes_are_prefix_stable():
    ids = list(range(64))
    h_full = block_hashes(ids, block_size=16)
    h_prefix = block_hashes(ids[:32], block_size=16)
    # A shared token prefix must produce identical leading block hashes.
    assert h_full[:2] == h_prefix


def test_block_hashes_diverge_on_any_token_change():
    ids = list(range(64))
    changed = list(ids)
    changed[5] = 999  # change one token in block 0
    assert block_hashes(ids, 16)[0] != block_hashes(changed, 16)[0]
    # later blocks also diverge because the hash is chained
    assert block_hashes(ids, 16)[2] != block_hashes(changed, 16)[2]


def test_radix_match_extends_with_shared_prefix():
    tree = RadixTree(block_size=16)
    hashes = block_hashes(list(range(64)), 16)
    assert tree.match(hashes) == 0  # empty tree
    # insert manually via simulator semantics
    node = tree.root
    for depth, h in enumerate(hashes):
        child = type(node)(block_hash=h, depth=depth + 1, parent=node)
        child.cached = True
        node.children[h] = child
        node = child
    assert tree.match(hashes) == len(hashes)
    assert tree.match(hashes[:2]) == 2


def test_identical_requests_fully_cache(clean_trace):
    tok = FallbackTokenizer()
    streams = [tokenize_request(r, tok) for r in clean_trace]
    sim = Simulator(block_size=16)
    results = sim.run(streams)
    # first request warms the cache; later ones with the same question hit heavily
    assert results[0].cached_blocks == 0
    assert sum(r.cached_blocks for r in results[1:]) > 0


def test_eviction_caps_residency():
    from cachetrace.core.model import TokenStream

    # 10 fully-distinct requests, cache capped at 8 blocks.
    streams = [
        TokenStream(request_id=f"r{i}", token_ids=[i * 1000 + j for j in range(64)])
        for i in range(10)
    ]
    sim = Simulator(block_size=16, cache_blocks=8)
    sim.run(streams)
    assert sim.tree.resident_blocks <= 8
