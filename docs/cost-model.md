# Cost model

cachetrace turns recoverable prefill tokens into dollars two ways.

## Self-hosted (default)

Waste per token is GPU time:

```
$/token = ($/GPU-hour / 3600) / (prefill_tokens_per_sec * engine_factor)
```

- `prefill_tokens_per_sec` and `$/GPU-hour` come from `core/cost/catalog/gpus.yaml`
  (defaults assume a ~7-8B dense model in bf16 on typical cloud on-demand pricing).
- `engine_factor` scales throughput per engine (`engines.yaml`): vLLM = 1.0 baseline,
  SGLang slightly higher on prefix-heavy loads, TGI slightly lower.

Override anything: `--gpu`, `--engine`, `--price-per-gpu-hour`, or `--price-per-token`.

## Hosted providers

With `--provider openai|anthropic|google|deepseek`, waste per token is the gap between
the normal input price and the cache-read price (from `providers.yaml`): those tokens
were billed at full price instead of the discounted cache-read rate.

```
waste_per_token = (input_price - cache_read_price)   # per token
```

## Monthly projection

```
per_request_recoverable = recoverable_tokens_in_trace / n_requests
monthly_waste = per_request_recoverable * monthly_volume * $/token
```

Set your traffic with `--volume`. All catalog numbers are illustrative and change
often; override them for anything you plan to act on. The **relative** gap (actual vs
achievable) is robust even when the absolute dollar figure is approximate.
