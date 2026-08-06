"""JSON serialization of audit/fix results for CI and programmatic use."""

from __future__ import annotations

from typing import Any

from cachetrace.core.analyze import AuditReport
from cachetrace.core.fix import FixPlan


def audit_to_dict(report: AuditReport) -> dict[str, Any]:
    return {
        "n_requests": report.n_requests,
        "model": report.model,
        "tokenizer": report.tokenizer_name,
        "approximate": report.approximate,
        "actual_hit_rate": report.actual.token_hit_rate,
        "achievable_hit_rate": report.achievable.token_hit_rate,
        "gap": report.gap,
        "recoverable_tokens": report.recoverable_tokens,
        "busters": [
            {
                "cause": b.cause,
                "path": b.path,
                "detail": b.detail,
                "requests_busted": b.requests_busted,
                "requests_total": b.requests_total,
                "pct_requests_busted": b.pct_requests_busted,
                "wasted_tokens": b.wasted_tokens,
                "examples": b.examples,
            }
            for b in report.busters
        ],
        "cost": {
            "usd_per_token": report.cost.usd_per_token,
            "basis": report.cost.basis,
            "monthly_volume": report.cost.monthly_volume,
            "monthly_recoverable_tokens": report.cost.monthly_recoverable_tokens,
            "monthly_waste_usd": report.cost.monthly_waste_usd,
        },
    }


def fix_to_dict(plan: FixPlan) -> dict[str, Any]:
    return {
        "model": plan.model,
        "actual_hit_rate": plan.actual_hit_rate,
        "projected_hit_rate": plan.projected_hit_rate,
        "gain": plan.gain,
        "rules": [
            {
                "target": r.target,
                "role": r.role,
                "cause": r.cause,
                "action": r.action,
                "pattern": r.pattern,
                "replacement": r.replacement,
                "est_tokens_saved": r.est_tokens_saved,
                "detail": r.detail,
            }
            for r in plan.rules
        ],
    }
