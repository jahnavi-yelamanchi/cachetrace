"""Benchmark generators reproduce real busters and the runner reports them."""

from __future__ import annotations

from cachetrace.bench.frameworks import FRAMEWORKS
from cachetrace.bench.run import render_markdown, run_benchmark
from cachetrace.config import Config

CFG = Config(tokenizer="fallback", gpu="h100", monthly_volume=1_000_000)


def test_all_frameworks_generate_traces():
    for name, gen in FRAMEWORKS.items():
        reqs = gen()
        assert len(reqs) >= 2, name
        assert all(r.messages for r in reqs)


def test_benchmark_finds_a_gap_for_each_framework():
    results = run_benchmark(config=CFG)
    assert len(results) == len(FRAMEWORKS)
    for r in results:
        # every default template ships at least one buster -> a real gap
        assert r.achievable >= r.actual
        assert r.gap > 0
        assert r.top_cause != "none"


def test_expected_top_causes():
    by_key = {r.key: r for r in run_benchmark(config=CFG)}
    assert by_key["langchain-react"].top_cause == "timestamp"
    assert by_key["crewai"].top_cause == "uuid"
    assert by_key["openai-agents"].top_cause == "reordering"


def test_render_markdown_has_table():
    md = render_markdown(run_benchmark(config=CFG), CFG)
    assert "| Framework |" in md
    assert "LangChain ReAct agent" in md
