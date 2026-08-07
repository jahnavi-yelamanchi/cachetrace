"""Run the audit across framework prompt reconstructions and report the findings."""

from __future__ import annotations

from dataclasses import dataclass

from cachetrace.bench.frameworks import FRAMEWORKS, LABELS
from cachetrace.config import Config
from cachetrace.core.analyze import audit


@dataclass
class BenchResult:
    key: str
    label: str
    actual: float
    achievable: float
    top_cause: str
    top_location: str
    monthly_waste_usd: float
    n_requests: int

    @property
    def gap(self) -> float:
        return self.achievable - self.actual


def run_benchmark(names: list[str] | None = None, config: Config | None = None) -> list[BenchResult]:
    config = config or Config(tokenizer="fallback", gpu="h100", engine="vllm", monthly_volume=1_000_000)
    keys = names or list(FRAMEWORKS)
    results: list[BenchResult] = []
    for key in keys:
        gen = FRAMEWORKS.get(key)
        if gen is None:
            continue
        reqs = gen()
        report = audit(reqs, config)
        top = report.busters[0] if report.busters else None
        results.append(
            BenchResult(
                key=key,
                label=LABELS.get(key, key),
                actual=report.actual.token_hit_rate,
                achievable=report.achievable.token_hit_rate,
                top_cause=top.cause if top else "none",
                top_location=top.path if top else "-",
                monthly_waste_usd=report.cost.monthly_waste_usd,
                n_requests=report.n_requests,
            )
        )
    return results


def render_markdown(results: list[BenchResult], config: Config | None = None) -> str:
    config = config or Config(monthly_volume=1_000_000)
    lines = [
        "# cachetrace benchmark: wasted prefix cache in default agent prompts",
        "",
        "Each row reconstructs a framework's **default** prompt construction and runs "
        "`cachetrace audit`. Numbers use the dependency-free approximate tokenizer, an "
        f"H100 + vLLM cost model, and {config.monthly_volume:,} requests/month. Install "
        "`cachetrace[tiktoken]` for exact token counts.",
        "",
        "| Framework | Actual hit rate | Achievable | Gap | Top cache-buster | Est. waste/mo |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {r.actual:.0%} | {r.achievable:.0%} | **+{r.gap:.0%}** | "
            f"{r.top_cause} in `{r.top_location}` | ${r.monthly_waste_usd:,.0f} |"
        )
    lines += [
        "",
        "Run `cachetrace fix` on any of these to get the rewrite that closes the gap. "
        "Reproduce with `cachetrace bench`.",
        "",
    ]
    return "\n".join(lines)
