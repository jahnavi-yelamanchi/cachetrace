# cachetrace

**Prefix-Cache Efficiency Auditor.** Find and fix what silently busts your LLM prefix cache.

```bash
pip install cachetrace
cachetrace demo
```

Point it at a JSONL of production requests and it reports the actual vs achievable
prefix-cache hit rate, the top cache-busters by root cause, and the estimated monthly
dollar waste, then emits the fix.

- [Concepts](concepts.md): how the radix simulation, attribution, and no-overclaim fix work
- [Providers](providers.md): supported input formats and how to add one
- [Cost model](cost-model.md): how token waste becomes dollars
- [Benchmark](benchmark.md): wasted cache in popular agent frameworks' default prompts

See the [README](https://github.com/jahnavi-yelamanchi/cachetrace) for full CLI usage.
