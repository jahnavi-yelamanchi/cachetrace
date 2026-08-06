"""cachetrace — Prefix-Cache Efficiency Auditor.

Point it at a JSONL of production requests (or run it as a proxy) and it tells you
your real prefix-cache hit rate, the achievable rate with better ordering, the top
cache-busters by root cause, the monthly dollars you're burning — then rewrites your
prompt template so you stop.
"""

from cachetrace.core.model import (
    Message,
    Request,
    RequestBatch,
    ToolSpec,
)

__version__ = "0.1.0"

__all__ = [
    "Message",
    "Request",
    "RequestBatch",
    "ToolSpec",
    "__version__",
]
