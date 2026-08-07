"""Provider ingest adapters normalize to the canonical Request model."""

from __future__ import annotations

from cachetrace.core.ingest import detect_provider, from_dict


def test_gemini_detect_and_normalize():
    obj = {
        "model": "gemini-1.5-pro",
        "systemInstruction": {"parts": [{"text": "You are helpful"}]},
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{"functionDeclarations": [{"name": "lookup", "parameters": {"type": "object"}}]}],
    }
    assert detect_provider(obj) == "gemini"
    req = from_dict(obj, request_id="r1")
    assert req.messages[0].role == "system"
    assert req.messages[0].text() == "You are helpful"
    assert req.messages[1].role == "user" and req.messages[1].text() == "hi"
    assert req.tools[0].name == "lookup"


def test_bedrock_detect_and_normalize():
    obj = {
        "modelId": "anthropic.claude-3-sonnet",
        "system": [{"text": "Be terse"}],
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "toolConfig": {"tools": [{"toolSpec": {"name": "search", "inputSchema": {"json": {"type": "object"}}}}]},
    }
    assert detect_provider(obj) == "bedrock"
    req = from_dict(obj, request_id="r1")
    assert req.messages[0].text() == "Be terse"
    assert req.messages[1].text() == "hello"
    assert req.tools[0].name == "search"


def test_agent_framework_envelope():
    obj = {"kwargs": {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}], "model": "gpt-4o"}}
    assert detect_provider(obj) == "agent"
    req = from_dict(obj, request_id="r1")
    assert [m.role for m in req.messages] == ["system", "user"]
    assert req.model == "gpt-4o"


def test_agent_framework_flat_prompt():
    obj = {"prompt": "Answer the question. Thought:"}
    assert detect_provider(obj) == "agent"
    req = from_dict(obj, request_id="r1")
    assert req.messages[0].text().startswith("Answer the question")


def test_vllm_log_unwrap():
    obj = {"timestamp": 123, "request": {"model": "meta/llama", "messages": [{"role": "user", "content": "yo"}]}}
    req = from_dict(obj, request_id="r1", provider="vllm")
    assert req.messages[0].text() == "yo"
    assert req.arrival_ts == 123
