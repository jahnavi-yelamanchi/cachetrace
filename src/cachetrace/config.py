"""Runtime configuration for an audit/fix run.

Values come from CLI flags, a ``cachetrace.toml``/YAML file, or sensible defaults.
The config captures everything the radix simulator and cost model need to match a
real deployment.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    # --- model / tokenization ---
    model: str | None = None
    """Model id used to pick a tokenizer + chat template (e.g. 'gpt-4o',
    'meta-llama/Llama-3.1-8B-Instruct'). None → auto-detect from the trace."""

    tokenizer: str | None = None
    """Force a specific tokenizer backend: 'auto' | 'tiktoken' | 'hf' | 'fallback'."""

    # --- engine / radix simulation ---
    engine: str = "vllm"
    """Serving engine being modeled: 'vllm' | 'sglang' | 'tgi'."""

    block_size: int = 16
    """KV block size in tokens (vLLM default 16, SGLang 1)."""

    cache_blocks: int | None = None
    """KV cache capacity in blocks. None → unbounded (no eviction)."""

    cache_gb: float | None = None
    """Alternative to cache_blocks: KV cache size in GiB (converted via the GPU model)."""

    # --- cost model ---
    gpu: str = "h100"
    """GPU for the prefill cost model: 'a100' | 'h100' | 'l40s' | ..."""

    price_per_gpu_hour: float | None = None
    """Override $/GPU-hour. None → catalog default for the GPU."""

    price_per_token: float | None = None
    """Override $ per uncached prefill token. None → derived from GPU + engine."""

    monthly_volume: int = 1_000_000
    """Requests/month, used to project monthly dollar waste."""

    provider: str | None = None
    """Provider for hosted cached/uncached token pricing (overrides GPU cost model)."""

    @classmethod
    def load(cls, path: Path | None) -> Config:
        if path is None:
            for candidate in (Path("cachetrace.toml"), Path("cachetrace.yaml"), Path("cachetrace.yml")):
                if candidate.exists():
                    path = candidate
                    break
        if path is None or not Path(path).exists():
            return cls()
        raw = Path(path).read_text(encoding="utf-8")
        if str(path).endswith(".toml"):
            import tomllib

            data = tomllib.loads(raw)
        else:
            data = yaml.safe_load(raw) or {}
        return cls(**(data.get("cachetrace", data)))


DEFAULT_CONFIG = Config()
