"""Self-contained interactive HTML report / dashboard page.

One string, no external hosts: inline CSS + inline SVG bar charts. Safe to open
from disk, email, or serve from the dashboard. Reused by ``audit --html``,
``cachetrace serve``, and ``cachetrace bench``.
"""

from __future__ import annotations

import html

from cachetrace.core.analyze import AuditReport
from cachetrace.core.fix import FixPlan

_CAUSE_COLOR = {
    "timestamp": "#e8833a",
    "uuid": "#3a86e8",
    "random_id": "#8b5cf6",
    "counter": "#e8c53a",
    "reordering": "#22c1a4",
    "nondeterministic_json": "#22c1a4",
    "variable_content": "#9aa0a6",
}


def _esc(s: object) -> str:
    return html.escape(str(s))


def _bar(pct: float, color: str) -> str:
    w = max(0.0, min(100.0, pct * 100))
    return (
        f'<svg width="100%" height="22" role="img">'
        f'<rect x="0" y="0" width="100%" height="22" rx="4" fill="#26292e"/>'
        f'<rect x="0" y="0" width="{w:.1f}%" height="22" rx="4" fill="{color}"/>'
        f"</svg>"
    )


def _buster_rows(report: AuditReport) -> str:
    if not report.busters:
        return '<tr><td colspan="5">No cache-busters found.</td></tr>'
    max_tok = max((b.wasted_tokens for b in report.busters), default=1) or 1
    rows = []
    for b in report.busters:
        color = _CAUSE_COLOR.get(b.cause, "#9aa0a6")
        rows.append(
            "<tr>"
            f'<td><span class="dot" style="background:{color}"></span>{_esc(b.cause)}</td>'
            f"<td>{_esc(b.path)}</td>"
            f"<td>{b.pct_requests_busted * 100:.0f}%</td>"
            f'<td class="num">{b.wasted_tokens:,}</td>'
            f'<td style="width:30%">{_bar(b.wasted_tokens / max_tok, color)}</td>'
            "</tr>"
        )
    return "".join(rows)


def _rule_rows(plan: FixPlan | None) -> str:
    if plan is None or not plan.rules:
        return ""
    rows = []
    for r in plan.rules:
        color = _CAUSE_COLOR.get(r.cause, "#9aa0a6")
        rows.append(
            "<tr>"
            f'<td><span class="dot" style="background:{color}"></span>{_esc(r.cause)}</td>'
            f"<td>{_esc(r.target)}</td><td>{_esc(r.action)}</td>"
            f'<td class="num">{r.est_tokens_saved:,}</td>'
            f"<td>{_esc(r.detail)}</td></tr>"
        )
    verified = "verified ✓" if plan.verified else "projected"
    fix_block = f"""
    <section class="card">
      <h2>Fix <span class="pill green">{plan.projected_hit_rate:.0%} {verified}</span>
        <span class="muted">was {plan.actual_hit_rate:.0%}</span></h2>
      <table><thead><tr><th>Cause</th><th>Target</th><th>Action</th>
        <th class="num">Tokens saved</th><th>Detail</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table>
    </section>"""
    return fix_block


def render_html(report: AuditReport, plan: FixPlan | None = None, *, title: str = "cachetrace report") -> str:
    cost = report.cost
    approx = " (approx)" if report.approximate else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#16181c; color:#e6e6e6;
    font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:28px 32px; border-bottom:1px solid #2a2d33;
    background:linear-gradient(180deg,#1c1f24,#16181c); }}
  h1 {{ margin:0; font-size:22px; }} h1 .tt {{ color:#22c1a4; }}
  .sub {{ color:#9aa0a6; margin-top:4px; font-size:13px; }}
  main {{ max-width:960px; margin:0 auto; padding:24px 20px 60px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:14px; }}
  .stat {{ background:#1c1f24; border:1px solid #2a2d33; border-radius:10px; padding:16px; }}
  .stat .k {{ color:#9aa0a6; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  .stat .v {{ font-size:26px; font-weight:600; margin-top:6px; }}
  .v.red {{ color:#e5534b; }} .v.green {{ color:#22c1a4; }} .v.amber {{ color:#e8c53a; }}
  .card {{ background:#1c1f24; border:1px solid #2a2d33; border-radius:10px;
    padding:18px 20px; margin-top:22px; overflow-x:auto; }}
  h2 {{ margin:0 0 14px; font-size:16px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #24272c; white-space:nowrap; }}
  th {{ color:#9aa0a6; font-weight:500; }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; }}
  .pill {{ font-size:12px; padding:2px 8px; border-radius:20px; margin-left:8px; }}
  .pill.green {{ background:#12332b; color:#22c1a4; }}
  .muted {{ color:#9aa0a6; font-size:12px; margin-left:8px; }}
  footer {{ color:#6b7178; text-align:center; padding:24px; font-size:12px; }}
</style></head>
<body>
<header>
  <h1><span class="tt">cachetrace</span> report</h1>
  <div class="sub">{report.n_requests:,} requests · {_esc(report.model or "unknown model")} · {_esc(report.tokenizer_name)}{approx}</div>
</header>
<main>
  <div class="grid">
    <div class="stat"><div class="k">Actual hit rate</div><div class="v red">{report.actual.token_hit_rate:.1%}</div></div>
    <div class="stat"><div class="k">Achievable</div><div class="v green">{report.achievable.token_hit_rate:.1%}</div></div>
    <div class="stat"><div class="k">Recoverable gap</div><div class="v amber">+{report.gap:.1%}</div></div>
    <div class="stat"><div class="k">Monthly waste</div><div class="v red">${cost.monthly_waste_usd:,.0f}</div></div>
  </div>

  <section class="card">
    <h2>Top cache-busters</h2>
    <table><thead><tr><th>Root cause</th><th>Where</th><th>Requests busted</th>
      <th class="num">Tokens wasted</th><th></th></tr></thead>
      <tbody>{_buster_rows(report)}</tbody></table>
  </section>

  <section class="card">
    <h2>Cost model</h2>
    <div class="sub">{_esc(cost.basis)}</div>
    <div class="sub">{cost.monthly_recoverable_tokens:,.0f} recoverable tokens/mo at {cost.monthly_volume:,} req/mo
      → <b>${cost.monthly_waste_usd:,.2f}/month</b></div>
  </section>
  {_rule_rows(plan)}
</main>
<footer>generated by cachetrace</footer>
</body></html>"""
