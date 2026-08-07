"""cachetrace command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from cachetrace import __version__
from cachetrace.config import Config
from cachetrace.core.analyze import AuditReport, audit
from cachetrace.core.ingest import load_batch
from cachetrace.core.model import Request

app = typer.Typer(
    add_completion=False,
    help="Prefix-Cache Efficiency Auditor — find and fix what busts your LLM prefix cache.",
    no_args_is_help=True,
)
console = Console()

# ---- shared options -------------------------------------------------------

ModelOpt = typer.Option(None, "--model", "-m", help="Model id (picks tokenizer + chat template).")
TokenizerOpt = typer.Option(None, "--tokenizer", help="Force backend: auto|tiktoken|hf|fallback.")
EngineOpt = typer.Option("vllm", "--engine", help="Serving engine: vllm|sglang|tgi.")
GpuOpt = typer.Option("h100", "--gpu", help="GPU for the cost model: h100|a100|l40s|...")
BlockOpt = typer.Option(16, "--block-size", help="KV block size in tokens.")
CacheBlocksOpt = typer.Option(None, "--cache-blocks", help="KV cache capacity in blocks (default: unbounded).")
VolumeOpt = typer.Option(1_000_000, "--volume", help="Requests/month for the cost projection.")
PriceTokenOpt = typer.Option(None, "--price-per-token", help="Override $ per uncached prefill token.")
PriceHourOpt = typer.Option(None, "--price-per-gpu-hour", help="Override $/GPU-hour.")
ProviderOpt = typer.Option(None, "--provider", help="Hosted provider pricing: openai|anthropic|google|deepseek.")
LimitOpt = typer.Option(None, "--limit", help="Only read the first N requests.")
ConfigOpt = typer.Option(None, "--config", "-c", help="Path to cachetrace.toml/yaml.")


def _build_config(**kw) -> Config:
    base = Config.load(kw.pop("config", None))
    data = base.model_dump()
    mapping = {
        "model": "model", "tokenizer": "tokenizer", "engine": "engine", "gpu": "gpu",
        "block_size": "block_size", "cache_blocks": "cache_blocks", "volume": "monthly_volume",
        "price_per_token": "price_per_token", "price_per_gpu_hour": "price_per_gpu_hour",
        "provider": "provider",
    }
    for arg, field in mapping.items():
        val = kw.get(arg)
        if val is not None:
            data[field] = val
    return Config(**data)


def _load(path: Path, provider: str | None, limit: int | None) -> list[Request]:
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(2)
    reqs = load_batch(path, provider=provider, limit=limit).requests
    if not reqs:
        console.print(f"[red]No requests parsed from[/red] {path}")
        raise typer.Exit(2)
    return reqs


def _run_audit(path: Path, cfg: Config, provider: str | None, limit: int | None) -> AuditReport:
    reqs = _load(path, provider, limit)
    return audit(reqs, cfg)


# ---- commands -------------------------------------------------------------


@app.command(name="audit")
def audit_cmd(
    file: Path = typer.Argument(..., help="JSONL trace of requests."),
    model: str | None = ModelOpt,
    tokenizer: str | None = TokenizerOpt,
    engine: str = EngineOpt,
    gpu: str = GpuOpt,
    block_size: int = BlockOpt,
    cache_blocks: int | None = CacheBlocksOpt,
    volume: int = VolumeOpt,
    price_per_token: float | None = PriceTokenOpt,
    price_per_gpu_hour: float | None = PriceHourOpt,
    provider: str | None = ProviderOpt,
    limit: int | None = LimitOpt,
    config: Path | None = ConfigOpt,
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    html_out: Path | None = typer.Option(None, "--html", help="Write a self-contained HTML report."),
) -> None:
    """Report actual vs achievable prefix-cache hit rate, cache-busters, and $ waste."""
    cfg = _build_config(
        config=config, model=model, tokenizer=tokenizer, engine=engine, gpu=gpu,
        block_size=block_size, cache_blocks=cache_blocks, volume=volume,
        price_per_token=price_per_token, price_per_gpu_hour=price_per_gpu_hour, provider=provider,
    )
    reqs = _load(file, provider, limit)
    report = audit(reqs, cfg)
    if as_json:
        from cachetrace.report.json_ import audit_to_dict

        console.print_json(json.dumps(audit_to_dict(report)))
    else:
        from cachetrace.report.terminal import render_audit

        render_audit(report, console)
    if html_out:
        from cachetrace.core.fix import build_fix_plan
        from cachetrace.report.html import render_html

        plan = build_fix_plan(report, reqs)
        html_out.write_text(render_html(report, plan), encoding="utf-8")
        console.print(f"[green]wrote[/green] {html_out}")


@app.command()
def fix(
    file: Path = typer.Argument(..., help="JSONL trace of requests."),
    model: str | None = ModelOpt,
    tokenizer: str | None = TokenizerOpt,
    engine: str = EngineOpt,
    gpu: str = GpuOpt,
    block_size: int = BlockOpt,
    cache_blocks: int | None = CacheBlocksOpt,
    volume: int = VolumeOpt,
    price_per_token: float | None = PriceTokenOpt,
    price_per_gpu_hour: float | None = PriceHourOpt,
    provider: str | None = ProviderOpt,
    limit: int | None = LimitOpt,
    config: Path | None = ConfigOpt,
    out: Path | None = typer.Option(None, "--out", "-o", help="Directory to write fix artifacts."),
    emit: str = typer.Option("both", "--emit", help="What to write: python|yaml|both."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Emit the fix: canonicalization rules, a Python transform, and a template rewrite."""
    from cachetrace.core.fix import build_fix_plan
    from cachetrace.core.fix.emit import emit_python, emit_rewrite, emit_yaml

    cfg = _build_config(
        config=config, model=model, tokenizer=tokenizer, engine=engine, gpu=gpu,
        block_size=block_size, cache_blocks=cache_blocks, volume=volume,
        price_per_token=price_per_token, price_per_gpu_hour=price_per_gpu_hour, provider=provider,
    )
    reqs = _load(file, provider, limit)
    report = audit(reqs, cfg)
    plan = build_fix_plan(report, reqs)

    if as_json:
        from cachetrace.report.json_ import fix_to_dict

        console.print_json(json.dumps(fix_to_dict(plan)))
        return

    from cachetrace.report.terminal import render_fix

    render_fix(plan, console)
    console.print()
    console.print(emit_rewrite(plan))

    if out:
        out.mkdir(parents=True, exist_ok=True)
        written = []
        if emit in ("yaml", "both"):
            p = out / "cachetrace_fix.yaml"
            p.write_text(emit_yaml(plan), encoding="utf-8")
            written.append(p)
        if emit in ("python", "both"):
            p = out / "cachetrace_canonical.py"
            p.write_text(emit_python(plan), encoding="utf-8")
            written.append(p)
        console.print()
        for p in written:
            console.print(f"[green]wrote[/green] {p}")


@app.command()
def explain(
    file: Path = typer.Argument(..., help="JSONL trace of requests."),
    request_id: str = typer.Argument(..., help="Request id (or line index like 'req-3') to explain."),
    model: str | None = ModelOpt,
    tokenizer: str | None = TokenizerOpt,
    limit: int | None = LimitOpt,
    provider: str | None = ProviderOpt,
) -> None:
    """Show exactly where a single request diverges from its template family and why."""
    from cachetrace.core.attribute.family import cluster_families
    from cachetrace.core.attribute.report import analyze_family

    reqs = _load(file, provider, limit)
    target = next((r for r in reqs if r.id == request_id), None)
    if target is None:
        console.print(f"[red]No request with id[/red] {request_id}")
        raise typer.Exit(2)

    fam = next((f for f in cluster_families(reqs) if any(r.id == request_id for r in f.requests)), None)
    if fam is None or fam.size < 2:
        console.print("[yellow]This request has no siblings to compare against.[/yellow]")
        raise typer.Exit(0)

    report = analyze_family(fam)
    console.print(f"[bold]Request {request_id}[/bold] — family of {fam.size} requests")
    spans_by_path: dict[str, list] = {}
    for s in report.spans:
        spans_by_path.setdefault(s.path, []).append(s)
    from cachetrace.core.tokenize.render import render_segments

    for seg in render_segments(target):
        if seg.path.startswith("tools["):
            continue
        spans = spans_by_path.get(seg.path)
        if not spans:
            console.print(f"  [green]✓ cached[/green] {seg.path}")
            continue
        for s in spans:
            tag = "[red]✗ busts[/red]" if s.cls.normalizable else "[yellow]~ varies[/yellow]"
            console.print(f"  {tag} {seg.path} → [bold]{s.cls.cause}[/bold]: {s.cls.detail}")
    if spans_by_path.get("tools"):
        console.print("  [red]✗ busts[/red] tools → [bold]reordering[/bold]: tool list order varies")


@app.command()
def demo() -> None:
    """Run an end-to-end audit + fix on the bundled example trace."""
    example = Path(__file__).parent.parent / "examples" / "agent_trace.jsonl"
    console.print(f"[cyan]cachetrace demo[/cyan] — auditing bundled trace: {example.name}\n")
    cfg = _build_config(config=None, model="gpt-4o", tokenizer="fallback", engine="vllm", gpu="h100", volume=1_000_000)
    reqs = _load(example, None, None)
    report = audit(reqs, cfg)
    from cachetrace.report.terminal import render_audit, render_fix

    render_audit(report, console)
    console.print()
    from cachetrace.core.fix import build_fix_plan
    from cachetrace.core.fix.emit import emit_rewrite

    plan = build_fix_plan(report, reqs)
    render_fix(plan, console)
    console.print()
    console.print(emit_rewrite(plan))


@app.command()
def serve(
    trace: Path = typer.Option(..., "--trace", help="JSONL trace to visualize."),
    model: str | None = ModelOpt,
    tokenizer: str | None = TokenizerOpt,
    gpu: str = GpuOpt,
    engine: str = EngineOpt,
    provider: str | None = ProviderOpt,
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Serve an interactive web dashboard for a trace (needs cachetrace[web])."""
    try:
        import uvicorn

        from cachetrace.server.app import create_app
    except ImportError:
        console.print("[red]The dashboard needs the web extra:[/red] pip install 'cachetrace[web]'")
        raise typer.Exit(1) from None
    cfg = _build_config(config=None, model=model, tokenizer=tokenizer, gpu=gpu, engine=engine, provider=provider)
    app_ = create_app(trace, cfg)
    console.print(f"[cyan]cachetrace dashboard[/cyan] → http://{host}:{port}")
    uvicorn.run(app_, host=host, port=port, log_level="warning")


@app.command()
def proxy(
    upstream: str = typer.Option(..., "--upstream", help="Upstream base URL, e.g. https://api.openai.com"),
    record: Path = typer.Option("cachetrace_traffic.jsonl", "--record", help="JSONL sink for recorded traffic."),
    fix_rules: Path | None = typer.Option(None, "--fix-rules", help="Apply an emitted cachetrace_fix.yaml inline."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8100, "--port"),
) -> None:
    """Run a recording (and optionally fixing) OpenAI/Anthropic-compatible proxy."""
    try:
        import uvicorn

        from cachetrace.proxy.server import create_proxy
    except ImportError:
        console.print("[red]The proxy needs the proxy extra:[/red] pip install 'cachetrace[proxy]'")
        raise typer.Exit(1) from None
    rules = None
    if fix_rules:
        from cachetrace.core.fix.runtime import load_rules

        rules = load_rules(fix_rules)
    app_ = create_proxy(upstream, record, fix_rules=rules)
    tag = " (applying fix)" if rules else ""
    console.print(f"[cyan]cachetrace proxy{tag}[/cyan] http://{host}:{port} → {upstream}, recording to {record}")
    uvicorn.run(app_, host=host, port=port, log_level="warning")


@app.command()
def version() -> None:
    """Print the cachetrace version."""
    console.print(f"cachetrace {__version__}")


if __name__ == "__main__":
    app()
