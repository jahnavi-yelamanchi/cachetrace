"""Block-based radix tree over token sequences.

Mirrors how vLLM and SGLang key their prefix cache: tokens are grouped into
fixed-size blocks, each block is hashed together with the hash of its prefix
(a rolling/chained hash), and cache entries live at block granularity. Two
requests share cached prefill exactly up to their last common block — which is
why a single differing token near the front (a timestamp, a UUID) invalidates
everything after it.

This structure is deliberately simple and inspectable; the eviction dynamics
live in :mod:`cachetrace.core.radix.simulator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def block_hashes(token_ids: list[int], block_size: int) -> list[int]:
    """Chained per-block hashes. Block *i*'s hash depends on all tokens 0..i, so a
    prefix match on hashes is a true prefix match on tokens.

    Only full blocks are hashed (matching vLLM, which caches whole blocks); a
    trailing partial block is not cacheable and is ignored here.
    """
    hashes: list[int] = []
    parent = 0xCBF29CE484222325  # FNV offset basis (64-bit)
    n_full = len(token_ids) // block_size
    for b in range(n_full):
        h = parent
        for t in token_ids[b * block_size : (b + 1) * block_size]:
            h ^= (t + 1) & 0xFFFFFFFFFFFFFFFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        hashes.append(h)
        parent = h
    return hashes


@dataclass
class RadixNode:
    """A node keyed by a block hash. Depth = number of cached blocks to reach it."""

    block_hash: int
    depth: int
    parent: RadixNode | None = None
    children: dict[int, RadixNode] = field(default_factory=dict)
    # simulator bookkeeping
    last_access: int = 0
    cached: bool = False


class RadixTree:
    """Prefix tree of block hashes. Tracks which blocks are currently resident."""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.root = RadixNode(block_hash=0, depth=0)
        self.root.cached = True
        self.resident_blocks = 0

    def match(self, hashes: list[int]) -> int:
        """Return how many leading blocks of ``hashes`` are already resident."""
        node = self.root
        matched = 0
        for h in hashes:
            child = node.children.get(h)
            if child is None or not child.cached:
                break
            node = child
            matched += 1
        return matched

    def match_node(self, hashes: list[int]) -> tuple[RadixNode, int]:
        node = self.root
        matched = 0
        for h in hashes:
            child = node.children.get(h)
            if child is None or not child.cached:
                break
            node = child
            matched += 1
        return node, matched
