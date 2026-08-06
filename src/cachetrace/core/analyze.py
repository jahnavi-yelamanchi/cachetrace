"""Top-level audit pipeline: actual vs achievable hit rate, busters, and cost.

Ties the engine together:

1. Faithfully render + tokenize every request and simulate the *actual* cache.
2. Attribute recoverable waste to individual busters (ordered ablation).
3. Canonicalize every request by its family report and simulate the *achievable*
   cache — the ceiling you'd hit with better prompt construction.
4. Convert the actual→achievable token gap into a monthly dollar estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

from cachetrace.config import Config
from cachetrace.core.attribute import AttributionResult, CacheBuster, run_attribution
from cachetrace.core.attribute.report import FamilyReport
from cachetrace.core.cost.model import CostEstimate, estimate
from cachetrace.core.fix.canonicalize import canonicalize_request
from cachetrace.core.model import Request
from cachetrace.core.radix.metrics import HitStats, aggregate
from cachetrace.core.radix.simulator import Simulator
from cachetrace.core.tokenize.base import Tokenizer
from cachetrace.core.tokenize.registry import get_tokenizer
from cachetrace.core.tokenize.render import tokenize_request


@dataclass
class AuditReport:
    n_requests: int
    tokenizer_name: str
    approximate: bool
    model: str | None
    actual: HitStats
    achievable: HitStats
    busters: list[CacheBuster]
    reports: list[FamilyReport]
    cost: CostEstimate
    config: Config

    @property
    def recoverable_tokens(self) -> int:
        return max(0, self.actual.computed_tokens - self.achievable.computed_tokens)

    @property
    def gap(self) -> float:
        return self.achievable.token_hit_rate - self.actual.token_hit_rate


def _infer_model(requests: list[Request]) -> str | None:
    for r in requests:
        if r.model:
            return r.model
    return None


def _simulate(requests: list[Request], tok: Tokenizer, config: Config) -> HitStats:
    streams = [tokenize_request(r, tok) for r in requests]
    sim = Simulator(block_size=config.block_size, cache_blocks=config.cache_blocks)
    return aggregate(sim.run(streams))


def _canonicalize_all(requests: list[Request], attribution: AttributionResult) -> list[Request]:
    by_id: dict[int, Request] = {}
    for report in attribution.reports:
        for r in report.family.requests:
            by_id[id(r)] = canonicalize_request(r, report)
    return [by_id.get(id(r), r) for r in requests]


def audit(requests: list[Request], config: Config | None = None) -> AuditReport:
    config = config or Config()
    model = config.model or _infer_model(requests)
    tok = get_tokenizer(model, config.tokenizer or "auto")
    approximate = tok.name == "fallback"

    actual_stats = _simulate(requests, tok, config)
    attribution = run_attribution(requests, tok, config.block_size)
    achievable_stats = _simulate(_canonicalize_all(requests, attribution), tok, config)

    recoverable = max(0, actual_stats.computed_tokens - achievable_stats.computed_tokens)

    # Rescale per-buster waste so the breakdown sums to the global recoverable gap
    # (attribution runs isolated & unbounded; the headline uses the global sim).
    busters = attribution.busters
    if attribution.total_recoverable_tokens > 0 and recoverable > 0:
        factor = recoverable / attribution.total_recoverable_tokens
        for b in busters:
            b.wasted_tokens = round(b.wasted_tokens * factor)

    cost = estimate(recoverable, len(requests), config)

    return AuditReport(
        n_requests=len(requests),
        tokenizer_name=tok.name,
        approximate=approximate,
        model=model,
        actual=actual_stats,
        achievable=achievable_stats,
        busters=busters,
        reports=attribution.reports,
        cost=cost,
        config=config,
    )
