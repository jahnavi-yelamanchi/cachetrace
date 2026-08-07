# cachetrace benchmark: wasted prefix cache in default agent prompts

Each row reconstructs a framework's **default** prompt construction and runs `cachetrace audit`. Numbers use the dependency-free approximate tokenizer, an H100 + vLLM cost model, and 1,000,000 requests/month. Install `cachetrace[tiktoken]` for exact token counts.

| Framework | Actual hit rate | Achievable | Gap | Top cache-buster | Est. waste/mo |
|---|---:|---:|---:|---|---:|
| LangChain ReAct agent | 14% | 94% | **+80%** | timestamp in `system` | $29 |
| CrewAI crew | 22% | 92% | **+70%** | uuid in `system` | $26 |
| OpenAI Agents tool loop | 83% | 91% | **+8%** | reordering in `tools` | $3 |
| RAG chat | 0% | 85% | **+85%** | random_id in `system` | $10 |

Run `cachetrace fix` on any of these to get the rewrite that closes the gap. Reproduce with `cachetrace bench`.
