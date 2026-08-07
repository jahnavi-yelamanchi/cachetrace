"""Reproduce the *default* prompt construction of popular agent frameworks.

Each generator returns a list of canonical Requests that mirror how the framework
builds its prompt out of the box — including the cache-busting habits it ships
with. These are dependency-free reconstructions (no SDK, no network), faithful to
the documented default templates, so `cachetrace bench` can quantify the waste.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from cachetrace.core.model import Message, Request, ToolSpec

_TOOLS = [
    ToolSpec(name="search", description="search the web", parameters={"type": "object", "properties": {"q": {"type": "string"}}}),
    ToolSpec(name="calculator", description="do math", parameters={"type": "object", "properties": {"expr": {"type": "string"}}}),
    ToolSpec(name="lookup_docs", description="search internal docs", parameters={"type": "object", "properties": {"query": {"type": "string"}}}),
]
_QUESTIONS = [
    "What was our Q3 revenue?",
    "Summarize the latest incident report.",
    "How do I reset a user's password?",
    "What's the refund policy for enterprise?",
    "Compare plan A and plan B pricing.",
]
_N = 24


def _ts(i: int) -> str:
    return f"2026-08-06 {9 + i // 12:02d}:{(i * 5) % 60:02d}:{(i * 7) % 60:02d}"


def langchain_react() -> list[Request]:
    """LangChain ReAct agent: the default prompt stamps the current date/time into
    the system instructions."""
    reqs = []
    for i in range(_N):
        system = (
            "Answer the following questions as best you can. You have access to tools.\n"
            f"Current date and time: {_ts(i)}.\n"
            "Use the format: Thought / Action / Action Input / Observation. Begin!"
        )
        reqs.append(Request(id=f"lc-{i}", model="gpt-4o", provider="langchain",
            messages=[Message(role="system", content=system), Message(role="user", content=_QUESTIONS[i % len(_QUESTIONS)])],
            tools=list(_TOOLS)))
    return reqs


def crewai_crew() -> list[Request]:
    """CrewAI: each kickoff mints a fresh task id embedded in the agent's system
    prompt alongside the role/goal/backstory block."""
    reqs = []
    for i in range(_N):
        task_id = str(uuid.UUID(int=(i * 2654435761 + 12345) & ((1 << 128) - 1)))
        system = (
            "You are a Senior Research Analyst. Your goal is to uncover accurate insights. "
            "Backstory: you have 10 years of experience and never fabricate.\n"
            f"Task ID: {task_id}\n"
            "Answer the user's question using your tools."
        )
        reqs.append(Request(id=f"crew-{i}", model="gpt-4o", provider="crewai",
            messages=[Message(role="system", content=system), Message(role="user", content=_QUESTIONS[i % len(_QUESTIONS)])],
            tools=list(_TOOLS)))
    return reqs


def openai_agents() -> list[Request]:
    """OpenAI Agents-style tool loop: tools are gathered from a set, so their
    serialization order is non-deterministic across runs."""
    reqs = []
    for i in range(_N):
        system = (
            "You are a helpful assistant with tools. Call tools when needed and be concise. "
            "Follow the company policy at all times."
        )
        tools = list(_TOOLS)
        # emulate set-ordering churn: rotate the tool list every request
        rot = i % len(tools)
        tools = tools[rot:] + tools[:rot]
        reqs.append(Request(id=f"oa-{i}", model="gpt-4o", provider="openai-agents",
            messages=[Message(role="system", content=system), Message(role="user", content=_QUESTIONS[i % len(_QUESTIONS)])],
            tools=tools))
    return reqs


def rag_chat() -> list[Request]:
    """RAG chat: a per-request retrieval/trace id is prepended to the system prompt
    for logging, busting the otherwise-static instruction prefix."""
    reqs = []
    for i in range(_N):
        trace = f"{(i * 0x9E3779B97F4A7C15 + 0x1234) & 0xFFFFFFFFFFFFFFFF:016x}"
        system = (
            f"[retrieval_id={trace}] You are a documentation assistant. "
            "Answer only from the provided context. If unsure, say you don't know. "
            "Always cite the source section."
        )
        reqs.append(Request(id=f"rag-{i}", model="gpt-4o", provider="rag",
            messages=[Message(role="system", content=system), Message(role="user", content=_QUESTIONS[i % len(_QUESTIONS)])]))
    return reqs


FRAMEWORKS: dict[str, Callable[[], list[Request]]] = {
    "langchain-react": langchain_react,
    "crewai": crewai_crew,
    "openai-agents": openai_agents,
    "rag-chat": rag_chat,
}

LABELS = {
    "langchain-react": "LangChain ReAct agent",
    "crewai": "CrewAI crew",
    "openai-agents": "OpenAI Agents tool loop",
    "rag-chat": "RAG chat",
}
