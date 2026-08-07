# Providers

cachetrace normalizes every input into one canonical `Request`, so audit and fix work
the same regardless of source. The format is auto-detected per line; force it with
`--provider`.

| Provider / format | Detected by | Notes |
|---|---|---|
| OpenAI-compatible | `messages` | OpenAI, Azure, Together, Fireworks, Groq, DeepSeek, Mistral, and vLLM/SGLang/TGI OpenAI servers |
| Anthropic Messages | top-level `system`, or tools with `input_schema` | system lifted into the message list |
| Google Gemini / Vertex | `contents` / `systemInstruction` | `parts[].text`, `functionDeclarations` |
| AWS Bedrock Converse | `toolConfig` / `modelId` / content blocks | `system[].text`, `toolConfig.tools[].toolSpec` |
| Agent-framework traces | wrapper keys (`input`, `kwargs`, `request`) or flat `prompt` | LangChain / LlamaIndex-style loggers |
| vLLM / server logs | `request` / `body` wrapper | unwraps to the chat body |

## Input format

One JSON object per line (JSONL). A minimal OpenAI-style row:

```jsonl
{"model": "gpt-4o", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "tools": [...]}
```

Add `"arrival_ts"` (unix seconds) if you want arrival-order fidelity for eviction
modeling; otherwise file order is used.

## Adding a provider

Drop a module in `src/cachetrace/core/ingest/` exposing
`from_dict(obj, *, request_id, provider) -> Request` and register it in the adapter
table in `core/ingest/__init__.py`. See `gemini.py` for a compact example.
