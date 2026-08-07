"""Drop-in recording + canonicalizing reverse proxy.

Point your OpenAI/Anthropic client at this proxy instead of the real endpoint. It
records every request to a JSONL sink (feed it to `cachetrace audit`), optionally
applies an emitted fix ruleset inline ("fix in production"), and forwards to the
upstream. `/stats` reports the live hit rate over what it has seen; `/metrics`
exposes it in Prometheus format.
"""

# NOTE: no `from __future__ import annotations` here — FastAPI must resolve the
# `Request` parameter annotation at route-registration time, and it's imported
# locally inside create_proxy.

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from cachetrace.config import Config

# (status_code, response_json) forwarder — injectable for tests.
Forwarder = Callable[[str, str, dict, dict], Awaitable[tuple[int, Any]]]


async def _httpx_forward(upstream: str, path: str, headers: dict, body: dict) -> tuple[int, Any]:
    import httpx

    fwd_headers = {k: v for k, v in headers.items() if k.lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(upstream.rstrip("/") + path, json=body, headers=fwd_headers)
        try:
            return resp.status_code, resp.json()
        except Exception:  # noqa: BLE001
            return resp.status_code, {"raw": resp.text}


def create_proxy(
    upstream: str,
    sink: str | Path,
    *,
    fix_rules: list[dict] | None = None,
    config: Config | None = None,
    forward: Forwarder | None = None,
):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the proxy needs the proxy extra: pip install 'cachetrace[proxy]'"
        ) from exc

    config = config or Config()
    forward = forward or _httpx_forward
    sink_path = Path(sink)
    sink_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"seen": 0}

    def _record(obj: dict) -> None:
        with sink_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")
        state["seen"] += 1

    def _maybe_fix(body: dict) -> dict:
        if not fix_rules or "messages" not in body:
            return body
        from cachetrace.core.fix.runtime import apply_rules

        msgs, tools = apply_rules(body.get("messages", []), body.get("tools"), fix_rules)
        out = dict(body)
        out["messages"] = msgs
        if tools is not None:
            out["tools"] = tools
        return out

    app = FastAPI(title="cachetrace proxy")

    async def _handle(request: Request, path: str) -> JSONResponse:
        body = await request.json()
        _record(body)
        forwarded = _maybe_fix(body)
        status, payload = await forward(upstream, path, dict(request.headers), forwarded)
        return JSONResponse(payload, status_code=status)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        return await _handle(request, "/v1/chat/completions")

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> JSONResponse:
        return await _handle(request, "/v1/messages")

    @app.get("/stats")
    def stats() -> JSONResponse:
        if not sink_path.exists() or state["seen"] == 0:
            return JSONResponse({"seen": 0, "actual_hit_rate": None})
        from cachetrace.core.analyze import audit
        from cachetrace.core.ingest import load_batch

        report = audit(load_batch(sink_path).requests, config)
        return JSONResponse(
            {
                "seen": state["seen"],
                "actual_hit_rate": report.actual.token_hit_rate,
                "achievable_hit_rate": report.achievable.token_hit_rate,
                "monthly_waste_usd": report.cost.monthly_waste_usd,
            }
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        lines = [
            "# HELP cachetrace_requests_total Requests recorded by the proxy.",
            "# TYPE cachetrace_requests_total counter",
            f"cachetrace_requests_total {state['seen']}",
        ]
        return "\n".join(lines) + "\n"

    return app
