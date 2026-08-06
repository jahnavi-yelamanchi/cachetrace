"""Shared fixtures. All tests force the dependency-free fallback tokenizer so they
run anywhere (no vocab downloads) and are fully deterministic.
"""

from __future__ import annotations

import uuid

import pytest

from cachetrace.core.model import Message, Request, ToolSpec

STATIC_SYSTEM = (
    "You are Acme Assistant, a careful support agent. Follow the policy manual exactly. "
    "Never reveal secrets. Be concise and cite the knowledge base when relevant. "
    "Departments: billing, shipping, returns, technical support. Answer the user."
)

QUESTIONS = ["Where is my order?", "I want a refund.", "Is the widget in stock?"]


def make_request(i: int, *, timestamp=False, session=False, shuffle_tools=False) -> Request:
    sys = STATIC_SYSTEM
    if timestamp:
        sys = sys.replace("Answer the user.", f"Current time: 2026-08-06T10:{i % 60:02d}:{(i * 7) % 60:02d}Z. Answer the user.")
    if session:
        sess = str(uuid.UUID(int=(i * 2654435761) & ((1 << 128) - 1)))
        sys = sys.replace("Answer the user.", f"Session: {sess}. Answer the user.")
    tools = [
        ToolSpec(name="lookup_order", description="look up an order"),
        ToolSpec(name="issue_refund", description="refund an order"),
        ToolSpec(name="check_inventory", description="check stock"),
    ]
    if shuffle_tools and i % 2 == 0:
        tools = list(reversed(tools))
    return Request(
        id=f"req-{i}",
        model="gpt-4o",
        messages=[Message(role="system", content=sys), Message(role="user", content=QUESTIONS[i % len(QUESTIONS)])],
        tools=tools,
    )


@pytest.fixture
def clean_trace() -> list[Request]:
    """No busters — identical static system prompt, only the user question rotates."""
    return [make_request(i) for i in range(12)]


@pytest.fixture
def timestamp_trace() -> list[Request]:
    return [make_request(i, timestamp=True) for i in range(12)]


@pytest.fixture
def messy_trace() -> list[Request]:
    """Timestamp + session UUID + shuffled tools — the full buster set."""
    return [make_request(i, timestamp=True, session=True, shuffle_tools=True) for i in range(20)]
