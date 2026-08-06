"""Rich terminal rendering of an AuditReport and FixPlan."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cachetrace.core.analyze import AuditReport
from cachetrace.core.fix import FixPlan

_CAUSE_EMOJI = {
    "timestamp": "🕒",
    "uuid": "🔑",
    "random_id": "🎲",
    "counter": "🔢",
    "reordering": "🔀",
    "nondeterministic_json": "🔀",
    "variable_content": "📝",
}


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_audit(report: AuditReport, console: Console | None = None) -> None:
    console = console or Console()

    gap = report.gap
    bar_actual = _fmt_pct(report.actual.token_hit_rate)
    bar_ach = _fmt_pct(report.achievable.token_hit_rate)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(justify="right", style="bold")
    summary.add_column()
    summary.add_row("Requests", f"{report.n_requests:,}")
    summary.add_row("Model", report.model or "—")
    summary.add_row("Tokenizer", report.tokenizer_name + (" (approx)" if report.approximate else ""))
    summary.add_row("Actual hit rate", Text(bar_actual, style="red" if report.actual.token_hit_rate < 0.5 else "yellow"))
    summary.add_row("Achievable hit rate", Text(bar_ach, style="green"))
    summary.add_row("Recoverable gap", Text(f"+{_fmt_pct(gap)}", style="bold green"))
    summary.add_row("Recoverable tokens", f"{report.recoverable_tokens:,} in trace")
    console.print(Panel(summary, title="[bold]cachetrace audit[/bold]", border_style="cyan"))

    if report.busters:
        t = Table(title="Top cache-busters (by prefill tokens wasted)", header_style="bold magenta")
        t.add_column("Root cause")
        t.add_column("Where")
        t.add_column("Requests busted", justify="right")
        t.add_column("Tokens wasted", justify="right")
        t.add_column("Example")
        for b in report.busters:
            emoji = _CAUSE_EMOJI.get(b.cause, "•")
            t.add_row(
                f"{emoji} {b.cause}",
                b.path,
                _fmt_pct(b.pct_requests_busted),
                f"{b.wasted_tokens:,}",
                (b.examples[0][:32] if b.examples else ""),
            )
        console.print(t)
    else:
        console.print("[green]No cache-busters found — prefix caching is already efficient.[/green]")

    cost = report.cost
    cost_panel = Table.grid(padding=(0, 2))
    cost_panel.add_column(justify="right", style="bold")
    cost_panel.add_column()
    cost_panel.add_row("Basis", cost.basis)
    cost_panel.add_row("At volume", f"{cost.monthly_volume:,} req/mo")
    cost_panel.add_row("Wasted tokens", f"{cost.monthly_recoverable_tokens:,.0f}/mo")
    cost_panel.add_row("Estimated waste", Text(f"${cost.monthly_waste_usd:,.2f} / month", style="bold red"))
    console.print(Panel(cost_panel, title="[bold]💸 Cost of busted cache[/bold]", border_style="red"))

    if report.approximate:
        console.print(
            r"[dim]Note: approximate tokenizer in use. Install cachetrace\[tiktoken] or "
            r"cachetrace\[hf] for exact token counts.[/dim]"
        )
    console.print("[dim]Run [bold]cachetrace fix[/bold] to emit the rewrite + canonicalization rules.[/dim]")


def render_fix(plan: FixPlan, console: Console | None = None) -> None:
    console = console or Console()
    header = Text()
    label = "Verified hit rate " if plan.verified else "Projected hit rate "
    header.append(label, style="bold")
    header.append(f"{plan.projected_hit_rate:.0%}", style="bold green")
    header.append(f"  (was {plan.actual_hit_rate:.0%}, ", style="dim")
    header.append(f"+{plan.gain:.0%})", style="green")
    if plan.verified:
        header.append("  ✓ measured by applying the emitted rules", style="dim green")
    if plan.ceiling_hit_rate > plan.projected_hit_rate + 0.01:
        header.append(f"\nCeiling (full canonicalization): {plan.ceiling_hit_rate:.0%}", style="dim")
    console.print(Panel(header, title="[bold]cachetrace fix[/bold]", border_style="green"))

    if not plan.rules:
        console.print("[green]Nothing to fix — no normalizable cache-busters detected.[/green]")
        return

    t = Table(title="Canonicalization rules", header_style="bold magenta")
    t.add_column("Cause")
    t.add_column("Target")
    t.add_column("Action")
    t.add_column("Tokens saved", justify="right")
    t.add_column("Detail")
    for r in plan.rules:
        emoji = _CAUSE_EMOJI.get(r.cause, "•")
        action_style = {"sort": "cyan", "remove": "yellow", "hoist": "blue"}.get(r.action, "white")
        t.add_row(
            f"{emoji} {r.cause}", r.target, Text(r.action, style=action_style),
            f"{r.est_tokens_saved:,}", r.detail[:48],
        )
    console.print(t)
