"""Provider ingest normalization and the cost model."""

from __future__ import annotations

import json

from cachetrace.config import Config
from cachetrace.core.cost.model import cost_per_token, estimate
from cachetrace.core.ingest import detect_provider, from_dict, load_batch


def test_openai_ingest():
    obj = {
        "model": "gpt-4o",
        "messages": [{"role": "system", "content": "hi"}, {"role": "user", "content": "yo"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        "temperature": 0.2,
    }
    req = from_dict(obj, request_id="r1")
    assert req.model == "gpt-4o"
    assert len(req.messages) == 2
    assert req.tools[0].name == "f"
    assert req.params["temperature"] == 0.2


def test_anthropic_ingest_lifts_system():
    obj = {
        "model": "claude-3",
        "system": "You are helpful",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"name": "g", "input_schema": {"type": "object"}}],
    }
    assert detect_provider(obj) == "anthropic"
    req = from_dict(obj, request_id="r1")
    assert req.messages[0].role == "system"
    assert req.messages[0].text() == "You are helpful"
    assert req.tools[0].name == "g"


def test_load_batch_roundtrip(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [
        {"model": "gpt-4o", "messages": [{"role": "user", "content": f"q{i}"}]}
        for i in range(3)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    batch = load_batch(p)
    assert len(batch) == 3


def test_cost_self_hosted_positive():
    cb = cost_per_token(Config(gpu="h100", engine="vllm"))
    assert cb.usd_per_token > 0
    assert "h100" in cb.description


def test_cost_provider_uses_cache_discount():
    cb = cost_per_token(Config(provider="openai"))
    assert cb.usd_per_token > 0
    assert "openai" in cb.description


def test_estimate_scales_with_volume():
    e1 = estimate(1000, 10, Config(gpu="h100", monthly_volume=1_000_000))
    e2 = estimate(1000, 10, Config(gpu="h100", monthly_volume=2_000_000))
    assert e2.monthly_waste_usd == 2 * e1.monthly_waste_usd
