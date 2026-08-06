"""Fix engine — the hero.

Turns the audit's diagnosis into an actionable fix: a set of canonicalization
rules, a runnable Python transform, and a human-readable template rewrite. Every
recommendation is backed by re-simulation, so the projected hit rate is measured,
never asserted.

Rule of thumb for the recommended *action* per root cause:

* ``timestamp``          → **hoist**: move it out of the cached prefix (e.g. to a
  trailing message) so the static system prompt caches; the model still sees it.
* ``uuid/random_id/counter`` → **remove**: request/session/trace ids almost never
  affect the output and simply burst the prefix.
* ``reordering``         → **sort**: serialize fields/tools deterministically —
  fully semantics-preserving.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cachetrace.core.attribute.classify import (
    COUNTER,
    NONDETERMINISTIC_JSON,
    RANDOM_ID,
    REORDERING,
    TIMESTAMP,
    UUID,
)
from cachetrace.core.attribute.report import FamilyReport, SpanVariation

if TYPE_CHECKING:
    from cachetrace.core.analyze import AuditReport

_ACTION = {
    TIMESTAMP: "hoist",
    UUID: "remove",
    RANDOM_ID: "remove",
    COUNTER: "remove",
    REORDERING: "sort",
    NONDETERMINISTIC_JSON: "sort",
}


def _anchor(text: str, tail: bool) -> str:
    """A short, single-line anchor from the end/start of surrounding context."""
    text = text.replace("\n", " ")
    return text[-40:] if tail else text[:40]


@dataclass
class Rule:
    target: str          # 'system', 'messages[2]', 'tools'
    role: str | None  # which message role to apply to at runtime (or None)
    cause: str
    action: str          # 'hoist' | 'remove' | 'sort'
    pattern: str | None  # anchored regex locating the volatile span
    replacement: str     # what the matched span becomes (remove/hoist)
    canonical: str       # placeholder token shown in the rewrite
    detail: str
    example: str = ""
    est_tokens_saved: int = 0

    @property
    def title(self) -> str:
        return f"{self.cause} in {self.target}"


def _rule_from_variation(v: SpanVariation) -> Rule:
    action = _ACTION.get(v.cls.cause, "remove")
    role = "system" if v.path.startswith("system") else None
    if v.path == "tools":
        return Rule(
            target="tools", role=None, cause=v.cls.cause, action="sort",
            pattern=None, replacement="", canonical="",
            detail=v.cls.detail, example=", ".join(v.examples()[:2]),
        )
    left = _anchor(v.varying.left, tail=True)
    right = _anchor(v.varying.right, tail=False)
    pattern = re.escape(left) + r"(?P<vol>.*?)" + re.escape(right) if (left or right) else None
    replacement = left + right
    return Rule(
        target=v.path, role=role, cause=v.cls.cause, action=action,
        pattern=pattern, replacement=replacement, canonical=v.cls.canonical,
        detail=v.cls.detail, example=" | ".join(v.examples()[:2]),
    )


@dataclass
class FixPlan:
    rules: list[Rule] = field(default_factory=list)
    projected_hit_rate: float = 0.0
    actual_hit_rate: float = 0.0
    ceiling_hit_rate: float = 0.0  # best achievable (audit oracle)
    monthly_waste_usd: float = 0.0
    verified: bool = False  # projected was measured by applying the emitted rules
    model: str | None = None
    representative: FamilyReport | None = None

    @property
    def gain(self) -> float:
        return self.projected_hit_rate - self.actual_hit_rate


def build_fix_plan(
    report: AuditReport,
    requests: list | None = None,
) -> FixPlan:
    """Assemble a deduplicated FixPlan from an AuditReport.

    If ``requests`` (the original trace) is supplied, the projected hit rate is
    *measured* by applying the emitted rules and re-simulating — so the number
    shown is exactly what the emitted transform delivers, never an overclaim.
    """
    rules: dict[tuple[str, str], Rule] = {}
    for fam_report in report.reports:
        for v in fam_report.normalizable:
            rule = _rule_from_variation(v)
            key = (rule.target if rule.target == "tools" else rule.role or rule.target, rule.cause)
            rules.setdefault(key, rule)

    # Attach estimated savings from attribution busters.
    saved_by_key: dict[tuple[str, str], int] = {}
    for b in report.busters:
        k = ("tools" if b.path == "tools" else ("system" if b.path.startswith("system") else b.path), b.cause)
        saved_by_key[k] = saved_by_key.get(k, 0) + b.wasted_tokens
    for key, rule in rules.items():
        rule.est_tokens_saved = saved_by_key.get(key, 0)

    ordered = sorted(rules.values(), key=lambda r: r.est_tokens_saved, reverse=True)
    representative = max(report.reports, key=lambda r: r.family.size, default=None) if report.reports else None

    plan = FixPlan(
        rules=ordered,
        projected_hit_rate=report.achievable.token_hit_rate,
        actual_hit_rate=report.actual.token_hit_rate,
        ceiling_hit_rate=report.achievable.token_hit_rate,
        monthly_waste_usd=report.cost.monthly_waste_usd,
        model=report.model,
        representative=representative,
    )

    if requests:
        measured = _measure_projected(plan, requests, report.config)
        if measured is not None:
            plan.projected_hit_rate = measured
            plan.verified = True
    return plan


def apply_fix(requests: list, plan: FixPlan):
    """Apply the plan's emitted transform to canonical Requests, returning new ones.

    This runs the exact generated code, guaranteeing what we measure is what we ship.
    """
    from cachetrace.core.fix.emit import emit_python
    from cachetrace.core.ingest.openai import from_dict

    ns: dict = {}
    exec(emit_python(plan), ns)  # noqa: S102 — our own generated module
    canon = ns["canonicalize"]

    out = []
    for r in requests:
        msgs = []
        for m in r.messages:
            content = m.content if isinstance(m.content, str) else m.text()
            msgs.append({"role": m.role, "content": content})
        tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in r.tools
        ]
        new_msgs, new_tools = canon(msgs, tools)
        obj = {"model": r.model, "messages": new_msgs, "tools": new_tools}
        out.append(from_dict(obj, request_id=r.id, provider=r.provider or "openai"))
    return out


def _measure_projected(plan: FixPlan, requests: list, config) -> float | None:
    from cachetrace.core.analyze import audit

    try:
        fixed = apply_fix(requests, plan)
        return audit(fixed, config).actual.token_hit_rate
    except Exception:  # noqa: BLE001 — verification is best-effort; fall back to ceiling
        return None
