# Concepts

## Prefix caching, in one paragraph

Serving engines (vLLM, SGLang, TGI) cache the key/value tensors for prompt tokens
they have already processed. When a new request shares a leading run of tokens with
something still in the cache, the engine skips recomputing that prefix. Caching is
**prefix-only and token-exact**: the shared region must match from token zero, and a
single differing token invalidates everything after it. That is why one timestamp
near the top of a system prompt can bust the cache for the entire prompt.

## What cachetrace computes

1. **Render.** Each request is rendered to the exact token stream an engine sees,
   after chat templating, with a map from token ranges back to template positions
   (`system`, `tools[2]`, `messages[4]`).
2. **Simulate (actual).** A block-based radix tree replays the trace in arrival order
   with LRU eviction, mirroring how vLLM hashes 16-token blocks and how SGLang's
   RadixAttention shares prefixes. This yields the **actual** hit rate.
3. **Attribute.** Within a template family, each segment is decomposed into constant
   and varying pieces. Every varying piece is classified (timestamp, uuid, random id,
   counter, reordering, or genuine content) and blamed for the tokens it wastes, using
   ordered ablation so overlapping busters are counted honestly.
4. **Simulate (achievable).** The trace is re-rendered with volatile noise canonicalized
   and re-simulated to get the **achievable** ceiling.
5. **Cost.** The actual to achievable token gap is priced per GPU + engine (or per hosted
   provider's cache-read discount) and projected to your monthly volume.

## Template families

Requests only share a prefix if they came from the same template. cachetrace clusters
requests by a structural signature (message roles, tool-name set, segment layout) so a
support-bot prompt is never compared against a summarizer prompt. Clustering is
order-independent on tools, so a reordered tool list stays in one family and the
reordering itself is reported as a buster.

## The no-overclaim guarantee

`cachetrace fix` never asserts a number it cannot back up. It emits the canonicalization
rules, applies them to your trace, re-simulates, and reports the **measured** post-fix
hit rate. What you see is what the emitted transform actually delivers.
