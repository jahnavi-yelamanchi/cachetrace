"""Replay a request stream through the radix tree with realistic eviction.

The simulator processes requests in arrival order. For each request it:

1. hashes the token stream into blocks,
2. matches the longest already-resident prefix (a cache *hit* on those blocks),
3. inserts the remaining blocks, evicting least-recently-used leaf blocks first
   when the KV cache is over capacity — the same "evict leaves before their
   parents" rule real radix caches follow.

The per-request result records exactly how many prefill tokens were served from
cache vs recomputed, which rolls up into the actual hit rate.
"""

from __future__ import annotations

from dataclasses import dataclass

from cachetrace.core.model import TokenStream
from cachetrace.core.radix.tree import RadixNode, RadixTree, block_hashes


@dataclass
class RequestResult:
    request_id: str
    total_tokens: int
    total_blocks: int
    cached_blocks: int
    cached_tokens: int
    computed_tokens: int  # tokens actually prefilled (cache miss region)

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.total_tokens if self.total_tokens else 0.0


class Simulator:
    def __init__(self, block_size: int = 16, cache_blocks: int | None = None):
        self.block_size = block_size
        self.cache_blocks = cache_blocks  # None => unbounded
        self.tree = RadixTree(block_size)
        self.tick = 0
        self._cached_nodes: dict[int, RadixNode] = {}

    # -- eviction ---------------------------------------------------------
    def _is_leaf(self, node: RadixNode) -> bool:
        return not any(c.cached for c in node.children.values())

    def _evict_until_fit(self, protect: set[int]) -> None:
        if self.cache_blocks is None:
            return
        while self.tree.resident_blocks > self.cache_blocks:
            victim: RadixNode | None = None
            for node in self._cached_nodes.values():
                if id(node) in protect or not self._is_leaf(node):
                    continue
                if victim is None or node.last_access < victim.last_access:
                    victim = node
            if victim is None:
                break  # everything resident is protected this tick
            victim.cached = False
            self.tree.resident_blocks -= 1
            self._cached_nodes.pop(id(victim), None)

    # -- main step --------------------------------------------------------
    def step(self, stream: TokenStream) -> RequestResult:
        self.tick += 1
        hashes = block_hashes(stream.token_ids, self.block_size)
        total_blocks = len(hashes)
        total_tokens = len(stream.token_ids)

        node, matched = self.tree.match_node(hashes)
        # Touch the matched prefix so it stays warm (LRU).
        walk = node
        while walk is not None:
            walk.last_access = self.tick
            walk = walk.parent

        touched: set[int] = set()
        # Insert the remaining blocks, allocating cache for the miss region.
        for depth in range(matched, total_blocks):
            h = hashes[depth]
            child = node.children.get(h)
            if child is None:
                child = RadixNode(block_hash=h, depth=depth + 1, parent=node)
                node.children[h] = child
            if not child.cached:
                child.cached = True
                self.tree.resident_blocks += 1
                self._cached_nodes[id(child)] = child
            child.last_access = self.tick
            touched.add(id(child))
            node = child

        self._evict_until_fit(protect=touched)

        cached_tokens = matched * self.block_size
        return RequestResult(
            request_id=stream.request_id,
            total_tokens=total_tokens,
            total_blocks=total_blocks,
            cached_blocks=matched,
            cached_tokens=cached_tokens,
            computed_tokens=total_tokens - cached_tokens,
        )

    def run(self, streams: list[TokenStream]) -> list[RequestResult]:
        return [self.step(s) for s in streams]
