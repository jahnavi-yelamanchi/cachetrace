"""CLI-level tests via Typer's CliRunner."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from cachetrace.cli.main import app

runner = CliRunner()


def _trace(tmp_path, name, timestamped):
    p = tmp_path / name
    rows = []
    for i in range(8):
        sys = "You are a helpful assistant. Follow policy and be concise for the user."
        if timestamped:
            sys += f" time 2026-08-06T10:0{i}:00Z."
        rows.append({"model": "gpt-4o", "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": "hi"},
        ]})
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_demo_runs():
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "cachetrace" in result.output


def test_audit_json():
    import cachetrace

    example = f"{cachetrace.__path__[0]}/examples/agent_trace.jsonl"
    result = runner.invoke(app, ["audit", example, "--tokenizer", "fallback", "--json"])
    assert result.exit_code == 0
    assert '"actual_hit_rate"' in result.output


def test_diff_regression_exit_code(tmp_path):
    clean = _trace(tmp_path, "clean.jsonl", timestamped=False)
    busted = _trace(tmp_path, "busted.jsonl", timestamped=True)
    # candidate is worse than baseline -> should fail
    result = runner.invoke(app, ["diff", str(clean), str(busted), "--tokenizer", "fallback"])
    assert result.exit_code == 1
    # candidate better/equal -> should pass
    ok = runner.invoke(app, ["diff", str(busted), str(clean), "--tokenizer", "fallback"])
    assert ok.exit_code == 0


def test_init_writes_config(tmp_path):
    cfg = tmp_path / "cachetrace.toml"
    result = runner.invoke(app, ["init", "--path", str(cfg)])
    assert result.exit_code == 0
    assert cfg.exists()
    assert "cachetrace" in cfg.read_text()


def test_bench_runs(tmp_path):
    out = tmp_path / "BENCHMARK.md"
    result = runner.invoke(app, ["bench", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "| Framework |" in out.read_text()
