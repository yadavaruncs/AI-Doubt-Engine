"""LangGraph agent.

    route ──> out_of_scope ──────────────────┐
      │                                      │
      ├──> retrieve ──> conceptual ──────────┤──> END
      │                      │               │
      └──> retrieve ──> numeric ──> tools ───┘
                           ^           │
                           └───────────┘  (tool loop, bounded)

Routing first is what keeps cost down: an out-of-scope message never touches
the vector store, and a conceptual doubt never pays for the tool loop.
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import LLM_MODEL, ROUTER_MODEL
from app.llm import client, complete
from app.prompts import (
    ANSWER_SYSTEM,
    CONCEPTUAL_PROMPT,
    NUMERIC_PROMPT,
    OUT_OF_SCOPE_REPLY,
    ROUTER_PROMPT,
)
from app.retrieval import format_context, retrieve
from app.tools import TOOL_IMPLS, TOOL_SCHEMAS

MAX_TOOL_ROUNDS = 4


def _add(a: int, b: int) -> int:
    return a + b


class DoubtState(TypedDict, total=False):
    question: str
    route: Literal["conceptual", "numeric", "out_of_scope"]
    hits: list
    context: str
    answer: str
    sources: list[dict]
    tool_calls: list[dict]
    prompt_tokens: Annotated[int, _add]
    completion_tokens: Annotated[int, _add]
    llm_ms: Annotated[float, _add]


# --------------------------------------------------------------- nodes
def route_node(state: DoubtState) -> DoubtState:
    out = complete(
        ROUTER_PROMPT.format(question=state["question"]),
        model=ROUTER_MODEL,
        temperature=0.0,
        max_tokens=8,
    )
    label = out.text.strip().lower().strip('."\'')
    if label not in {"conceptual", "numeric", "out_of_scope"}:
        label = "conceptual"  # safest fallback: retrieve and explain
    return {
        "route": label,
        "prompt_tokens": out.prompt_tokens,
        "completion_tokens": out.completion_tokens,
        "llm_ms": out.latency_ms,
    }


def retrieve_node(state: DoubtState) -> DoubtState:
    hits = retrieve(state["question"])
    return {
        "hits": hits,
        "context": format_context(hits),
        "sources": [
            {"n": i, "source": h.source, "page": h.page, "score": round(h.score, 4)}
            for i, h in enumerate(hits, start=1)
        ],
    }


def conceptual_node(state: DoubtState) -> DoubtState:
    out = complete(
        CONCEPTUAL_PROMPT.format(
            context=state.get("context", ""), question=state["question"]
        ),
        system=ANSWER_SYSTEM,
    )
    return {
        "answer": out.text,
        "prompt_tokens": out.prompt_tokens,
        "completion_tokens": out.completion_tokens,
        "llm_ms": out.latency_ms,
    }


def numeric_node(state: DoubtState) -> DoubtState:
    """Native tool-calling loop. SymPy does the arithmetic; the LLM narrates."""
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": NUMERIC_PROMPT.format(
                context=state.get("context", ""), question=state["question"]
            ),
        },
    ]
    calls_made: list[dict] = []
    p_tokens = c_tokens = 0
    elapsed = 0.0

    for _ in range(MAX_TOOL_ROUNDS):
        started = time.perf_counter()
        resp = client().chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.1,
            max_tokens=900,
        )
        elapsed += (time.perf_counter() - started) * 1000
        usage = getattr(resp, "usage", None)
        p_tokens += getattr(usage, "prompt_tokens", 0) or 0
        c_tokens += getattr(usage, "completion_tokens", 0) or 0

        msg = resp.choices[0].message
        if not getattr(msg, "tool_calls", None):
            return {
                "answer": (msg.content or "").strip(),
                "tool_calls": calls_made,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "llm_ms": elapsed,
            }

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            fn = TOOL_IMPLS.get(tc.function.name)
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = (
                fn(**args)
                if fn
                else {"ok": False, "error": f"unknown tool {tc.function.name}"}
            )
            calls_made.append({"tool": tc.function.name, "args": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    return {
        "answer": "I couldn't finish this calculation reliably. Please rephrase the problem.",
        "tool_calls": calls_made,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "llm_ms": elapsed,
    }


def out_of_scope_node(state: DoubtState) -> DoubtState:
    return {"answer": OUT_OF_SCOPE_REPLY, "sources": [], "hits": []}


def _branch(state: DoubtState) -> str:
    return "reject" if state.get("route") == "out_of_scope" else "retriever"


def _after_retrieve(state: DoubtState) -> str:
    return "answer_numeric" if state.get("route") == "numeric" else "answer_conceptual"


def build_graph():
    # Node names must not collide with state keys ("route" is a key), hence
    # "router" / "retriever" / "answer_*".
    g = StateGraph(DoubtState)
    g.add_node("router", route_node)
    g.add_node("retriever", retrieve_node)
    g.add_node("answer_conceptual", conceptual_node)
    g.add_node("answer_numeric", numeric_node)
    g.add_node("reject", out_of_scope_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router", _branch, {"retriever": "retriever", "reject": "reject"}
    )
    g.add_conditional_edges(
        "retriever",
        _after_retrieve,
        {"answer_numeric": "answer_numeric", "answer_conceptual": "answer_conceptual"},
    )
    g.add_edge("answer_conceptual", END)
    g.add_edge("answer_numeric", END)
    g.add_edge("reject", END)
    return g.compile()


GRAPH = build_graph()
