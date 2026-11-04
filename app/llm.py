"""Thin OpenAI-compatible LLM client with per-call token and latency accounting.

Cost/latency per query is one of the JD's explicit responsibilities, so it is
measured here rather than guessed at.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import OpenAI

from app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT,
)

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY or "not-needed-for-local",
            timeout=LLM_TIMEOUT,
        )
    return _client


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str = LLM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> Completion:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    started = time.perf_counter()
    resp = client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    usage = getattr(resp, "usage", None)
    return Completion(
        text=(resp.choices[0].message.content or "").strip(),
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        latency_ms=latency_ms,
    )
