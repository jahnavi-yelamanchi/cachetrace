"""Dashboard + proxy: routes work and the proxy records/forwards (no network)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cachetrace.config import Config  # noqa: E402


def _write_trace(tmp_path):
    p = tmp_path / "trace.jsonl"
    rows = []
    for i in range(8):
        rows.append({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": f"You are a bot. time 2026-08-06T10:0{i}:00Z. Be nice and follow policy."},
                {"role": "user", "content": "hi"},
            ],
        })
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_dashboard_routes(tmp_path):
    from cachetrace.server.app import create_app

    app = create_app(_write_trace(tmp_path), Config(model="gpt-4o", tokenizer="fallback"))
    client = TestClient(app)

    assert client.get("/healthz").json()["requests"] == 8
    assert "<title>" in client.get("/").text
    audit = client.get("/api/audit").json()
    assert 0.0 <= audit["actual_hit_rate"] <= 1.0
    assert audit["achievable_hit_rate"] >= audit["actual_hit_rate"]
    assert "rules" in client.get("/api/fix").json()


def test_proxy_records_and_forwards(tmp_path):
    from cachetrace.proxy.server import create_proxy

    sink = tmp_path / "rec.jsonl"
    seen_upstream = {}

    async def fake_forward(upstream, path, headers, body):
        seen_upstream["path"] = path
        seen_upstream["body"] = body
        return 200, {"id": "resp-1", "choices": [{"message": {"content": "ok"}}]}

    app = create_proxy("https://upstream.test", sink, config=Config(tokenizer="fallback"), forward=fake_forward)
    client = TestClient(app)

    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.json()["id"] == "resp-1"
    assert seen_upstream["path"] == "/v1/chat/completions"

    # recorded to sink
    assert sink.exists()
    recorded = [json.loads(line) for line in sink.read_text().splitlines()]
    assert recorded[0]["messages"][0]["content"] == "hello"

    stats = client.get("/stats").json()
    assert stats["seen"] == 1
    assert "cachetrace_requests_total 1" in client.get("/metrics").text


def test_proxy_applies_fix_rules(tmp_path):
    from cachetrace.proxy.server import create_proxy

    captured = {}

    async def fake_forward(upstream, path, headers, body):
        captured["body"] = body
        return 200, {"ok": True}

    rules = [{"target": "tools", "role": None, "action": "sort", "pattern": None}]
    app = create_proxy("https://x.test", tmp_path / "r.jsonl", fix_rules=rules, forward=fake_forward)
    client = TestClient(app)
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"function": {"name": "zebra"}}, {"function": {"name": "apple"}}],
    }
    client.post("/v1/chat/completions", json=body)
    # tools were sorted by name before forwarding
    names = [t["function"]["name"] for t in captured["body"]["tools"]]
    assert names == ["apple", "zebra"]
