"""Local web dashboard: a read-only view of an audited trace.

`cachetrace serve --trace requests.jsonl` runs this. The page itself is the
self-contained HTML report; ``/api/audit`` and ``/api/fix`` expose the same data
as JSON for scripting.
"""

from __future__ import annotations

from pathlib import Path

from cachetrace.config import Config


def create_app(trace_path: str | Path, config: Config | None = None):
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the dashboard needs the web extra: pip install 'cachetrace[web]'"
        ) from exc

    from cachetrace.core.analyze import audit
    from cachetrace.core.fix import build_fix_plan
    from cachetrace.core.ingest import load_batch
    from cachetrace.report.html import render_html
    from cachetrace.report.json_ import audit_to_dict, fix_to_dict

    config = config or Config()
    requests = load_batch(trace_path).requests
    report = audit(requests, config)
    plan = build_fix_plan(report, requests)
    page = render_html(report, plan, title=f"cachetrace · {Path(trace_path).name}")

    app = FastAPI(title="cachetrace dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    @app.get("/api/audit")
    def api_audit() -> JSONResponse:
        return JSONResponse(audit_to_dict(report))

    @app.get("/api/fix")
    def api_fix() -> JSONResponse:
        return JSONResponse(fix_to_dict(plan))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "requests": len(requests)}

    return app
