"""Attribution: blame recoverable prefill tokens on specific cache-busters.

Per template family we compute how many tokens the actual (faithful) stream
recomputes versus a fully-canonicalized stream, then split that recoverable waste
across the individual busters by *ordered ablation* — normalizing them one at a
time in render order and crediting each with the tokens it frees. Fixing a buster
only helps once everything before it is also fixed, so ordered marginal
attribution is the honest accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cachetrace.core.attribute.family import cluster_families
from cachetrace.core.attribute.report import FamilyReport, analyze_family
from cachetrace.core.fix.canonicalize import canonicalize_request
from cachetrace.core.model import Request
from cachetrace.core.radix.simulator import Simulator
from cachetrace.core.tokenize.base import Tokenizer
from cachetrace.core.tokenize.render import tokenize_request

__all__ = ["CacheBuster", "AttributionResult", "run_attribution", "analyze_family"]


@dataclass
class CacheBuster:
    cause: str
    path: str
    detail: str
    wasted_tokens: int
    requests_busted: int
    requests_total: int
    examples: list[str] = field(default_factory=list)

    @property
    def pct_requests_busted(self) -> float:
        return self.requests_busted / self.requests_total if self.requests_total else 0.0


@dataclass
class AttributionResult:
    busters: list[CacheBuster]
    reports: list[FamilyReport]
    total_recoverable_tokens: int


def _computed_tokens(reqs: list[Request], tok: Tokenizer, block_size: int) -> int:
    """Isolated, unbounded simulation → total tokens recomputed (cache misses)."""
    sim = Simulator(block_size=block_size, cache_blocks=None)
    streams = [tokenize_request(r, tok) for r in reqs]
    return sum(r.computed_tokens for r in sim.run(streams))


def run_attribution(
    requests: list[Request],
    tok: Tokenizer,
    block_size: int = 16,
) -> AttributionResult:
    raw: list[CacheBuster] = []
    reports: list[FamilyReport] = []
    total_recoverable = 0

    for fam in cluster_families(requests):
        if fam.size < 2:
            continue
        report = analyze_family(fam)
        reports.append(report)
        norm = report.normalizable
        if not norm:
            continue

        actual_computed = _computed_tokens(fam.requests, tok, block_size)
        applied: set = set()
        prev = actual_computed

        # Ablate busters in render order; each marginal drop is its waste.
        for v in norm:
            applied.add(v.key)
            reqs_k = [canonicalize_request(r, report, apply=applied) for r in fam.requests]
            computed_k = _computed_tokens(reqs_k, tok, block_size)
            marginal = max(0, prev - computed_k)
            prev = computed_k
            total_recoverable += marginal
            raw.append(
                CacheBuster(
                    cause=v.cls.cause,
                    path=v.path,
                    detail=v.cls.detail,
                    wasted_tokens=marginal,
                    requests_busted=v.n_busted,
                    requests_total=fam.size,
                    examples=v.examples(),
                )
            )

    # Aggregate identical (cause, path) busters across families.
    merged: dict[tuple[str, str], CacheBuster] = {}
    for b in raw:
        key = (b.cause, b.path)
        if key in merged:
            m = merged[key]
            m.wasted_tokens += b.wasted_tokens
            m.requests_busted += b.requests_busted
            m.requests_total += b.requests_total
        else:
            merged[key] = b
    busters = sorted(merged.values(), key=lambda b: b.wasted_tokens, reverse=True)
    return AttributionResult(
        busters=busters, reports=reports, total_recoverable_tokens=total_recoverable
    )
