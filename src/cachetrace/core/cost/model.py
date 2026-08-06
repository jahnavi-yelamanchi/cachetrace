"""Translate recoverable prefill tokens into dollars.

Two modes:

* **Hosted provider** (``--provider openai|anthropic|...``): the waste per token is
  the gap between the normal input price and the cache-read price — those tokens
  were billed full price instead of as a cache read.
* **Self-hosted** (default): waste per token is GPU time. We derive
  ``$/token = ($/GPU-hour / 3600) / (prefill_tokens_per_sec * engine_factor)``.

Every number is overridable; the catalog defaults are labeled illustrative.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from cachetrace.config import Config

_CATALOG = Path(__file__).parent / "catalog"


@lru_cache(maxsize=4)
def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((_CATALOG / name).read_text(encoding="utf-8")) or {}


@dataclass
class CostBasis:
    usd_per_token: float
    description: str


def cost_per_token(config: Config) -> CostBasis:
    if config.price_per_token is not None:
        return CostBasis(config.price_per_token, "explicit --price-per-token")

    if config.provider:
        providers = _load("providers.yaml")
        p = providers.get(config.provider.lower())
        if p:
            waste = (p["input"] - p["cache_read"]) / 1_000_000
            return CostBasis(
                waste,
                f"{config.provider} input ${p['input']}/M vs cache-read ${p['cache_read']}/M",
            )

    gpus = _load("gpus.yaml")
    engines = _load("engines.yaml")
    gpu = gpus.get(config.gpu.lower(), gpus["h100"])
    price_hr = config.price_per_gpu_hour if config.price_per_gpu_hour is not None else gpu["price_per_hour"]
    tps = gpu["prefill_tokens_per_sec"] * engines.get(config.engine.lower(), 1.0)
    usd_per_token = (price_hr / 3600.0) / tps
    return CostBasis(
        usd_per_token,
        f"{config.gpu} @ ${price_hr}/hr, {config.engine} prefill ~{tps:.0f} tok/s",
    )


@dataclass
class CostEstimate:
    usd_per_token: float
    basis: str
    recoverable_tokens_per_request: float
    monthly_recoverable_tokens: float
    monthly_waste_usd: float
    monthly_volume: int


def estimate(recoverable_tokens_in_trace: float, n_requests: int, config: Config) -> CostEstimate:
    basis = cost_per_token(config)
    per_req = recoverable_tokens_in_trace / n_requests if n_requests else 0.0
    monthly_tokens = per_req * config.monthly_volume
    return CostEstimate(
        usd_per_token=basis.usd_per_token,
        basis=basis.description,
        recoverable_tokens_per_request=per_req,
        monthly_recoverable_tokens=monthly_tokens,
        monthly_waste_usd=monthly_tokens * basis.usd_per_token,
        monthly_volume=config.monthly_volume,
    )
