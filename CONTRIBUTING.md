# Contributing to cachetrace

Thanks for helping make LLM inference cheaper for everyone.

## Setup

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Project layout

```
src/cachetrace/
  core/
    ingest/     provider adapters -> canonical Request
    tokenize/   pluggable tokenizers + segment-aware renderer
    radix/      block-based radix tree + eviction simulator
    attribute/  token-diff decomposition, classifiers, waste attribution
    cost/       GPU/engine/provider cost catalog + model
    fix/        canonicalization, rule emission, verification
  cli/          the `cachetrace` command
  report/       terminal + JSON output
```

## Guidelines

- **Never overclaim.** Any projected/achievable number the tool reports must be
  produced by re-simulating the actual transformed stream. The `fix` round-trip test
  (`tests/test_audit_fix.py::test_fix_does_not_overclaim`) enforces this — keep it green.
- Add a golden fixture in `tests/conftest.py` for any new root-cause detector.
- Keep the base install light; heavy deps (transformers, fastapi) go behind extras.
- Run `ruff check` and `pytest` before opening a PR.

## Adding a provider adapter

Add a module under `core/ingest/` exposing `from_dict(obj, *, request_id, provider)`
that returns a canonical `Request`, then wire detection into `core/ingest/__init__.py`.

## Adding a root-cause detector

Add the pattern to `core/attribute/classify.py`, return a `Classification` with the
right `normalizable` flag and `canonical` placeholder, and add a golden test.
