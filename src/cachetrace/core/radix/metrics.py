"""Aggregate per-request simulator results into hit-rate statistics."""

from __future__ import annotations

from dataclasses import dataclass

from cachetrace.core.radix.simulator import RequestResult


@dataclass
class HitStats:
    n_requests: int
    total_tokens: int
    cached_tokens: int
    computed_tokens: int
    fully_cached_requests: int  # requests whose entire prefix was a hit

    @property
    def token_hit_rate(self) -> float:
        """Fraction of prompt tokens served from cache — the number engines report."""
        return self.cached_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def avg_prompt_tokens(self) -> float:
        return self.total_tokens / self.n_requests if self.n_requests else 0.0


def aggregate(results: list[RequestResult]) -> HitStats:
    total = sum(r.total_tokens for r in results)
    cached = sum(r.cached_tokens for r in results)
    computed = sum(r.computed_tokens for r in results)
    fully = sum(1 for r in results if r.total_blocks > 0 and r.cached_blocks == r.total_blocks)
    return HitStats(
        n_requests=len(results),
        total_tokens=total,
        cached_tokens=cached,
        computed_tokens=computed,
        fully_cached_requests=fully,
    )
