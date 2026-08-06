"""End-to-end audit + the critical no-overclaim guarantee for fix."""

from __future__ import annotations

from pathlib import Path

from cachetrace.config import Config
from cachetrace.core.analyze import audit
from cachetrace.core.fix import apply_fix, build_fix_plan
from cachetrace.core.ingest import load_batch

CFG = Config(model="gpt-4o", tokenizer="fallback", block_size=16, gpu="h100", monthly_volume=1_000_000)


def test_audit_reports_gap_on_messy_trace(messy_trace):
    rep = audit(messy_trace, CFG)
    assert rep.actual.token_hit_rate < rep.achievable.token_hit_rate
    assert rep.recoverable_tokens > 0
    assert rep.busters
    assert rep.cost.monthly_waste_usd > 0


def test_audit_clean_trace_has_small_gap(clean_trace):
    rep = audit(clean_trace, CFG)
    assert not rep.busters
    # nothing normalizable to recover
    assert rep.recoverable_tokens == 0


def test_achievable_never_below_actual(messy_trace):
    rep = audit(messy_trace, CFG)
    assert rep.achievable.token_hit_rate >= rep.actual.token_hit_rate


def test_fix_does_not_overclaim(messy_trace):
    """The verified projected rate must be achieved by actually applying the rules."""
    rep = audit(messy_trace, CFG)
    plan = build_fix_plan(rep, messy_trace)
    assert plan.verified
    # Re-apply the emitted transform independently and re-audit.
    fixed = apply_fix(messy_trace, plan)
    re_rep = audit(fixed, CFG)
    assert abs(re_rep.actual.token_hit_rate - plan.projected_hit_rate) < 1e-6
    # And it genuinely improves things.
    assert plan.projected_hit_rate > rep.actual.token_hit_rate


def test_fix_reaches_near_ceiling(messy_trace):
    rep = audit(messy_trace, CFG)
    plan = build_fix_plan(rep, messy_trace)
    # emitted rules should recover most of the achievable ceiling
    assert plan.projected_hit_rate >= 0.6 * plan.ceiling_hit_rate


def test_bundled_example_audits():
    example = Path(__file__).parent.parent / "src" / "cachetrace" / "examples" / "agent_trace.jsonl"
    reqs = load_batch(example).requests
    rep = audit(reqs, CFG)
    assert rep.n_requests == 40
    assert rep.busters
    plan = build_fix_plan(rep, reqs)
    assert plan.verified
    assert plan.projected_hit_rate > rep.actual.token_hit_rate
