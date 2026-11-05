"""FastAPI service.

POST /ask        -> answer a doubt
GET  /healthz    -> liveness + collection status
GET  /metrics    -> cache stats and rolling latency percentiles
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from statistics import median
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.cache import CACHE
from app.config import (
    ALLOW_UNAUTHENTICATED,
    API_KEY,
    COLLECTION,
    CORS_ORIGINS,
    RATE_LIMIT_PER_MIN,
)
from app.graph import GRAPH
from app.store import client, ensure_collection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not API_KEY and not ALLOW_UNAUTHENTICATED:
        raise RuntimeError(
            "API_KEY is empty, so /ask would be open. "
            "Set it, or ALLOW_UNAUTHENTICATED=1 for local use."
        )
    try:
        ensure_collection()
    except Exception as exc:  # noqa: BLE001 - don't crash-loop if Qdrant is slow to boot
        print(f"[startup] Qdrant not ready yet: {exc}")
    yield


app = FastAPI(title="AI Doubt Engine", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_LATENCIES: deque[float] = deque(maxlen=500)  # cache misses only, see ask()
_rate_lock = threading.Lock()
_RATE_MAX_IPS = 10_000
_rate_hits: dict[str, deque[float]] = {}  # client IP -> times of allowed requests


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default=[], max_length=6)
    use_cache: bool = True


class AskResponse(BaseModel):
    answer: str
    route: str | None = None
    sources: list[dict] = []
    tool_calls: list[dict] = []
    cached: bool = False
    cache_similarity: float | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@app.get("/healthz")
def healthz() -> dict:
    try:
        info = client().get_collection(COLLECTION)
        return {
            "status": "ok",
            "collection": COLLECTION,
            "points": info.points_count,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "error": str(exc)}


@app.get("/metrics")
def metrics() -> dict:
    lat = sorted(_LATENCIES)
    def pct(p: float) -> float:
        if not lat:
            return 0.0
        return round(lat[min(int(len(lat) * p), len(lat) - 1)], 2)

    return {
        "requests": len(lat),
        "latency_ms": {
            "p50": round(median(lat), 2) if lat else 0.0,
            "p95": pct(0.95),
            "p99": pct(0.99),
        },
        "cache": CACHE.stats(),
    }


def guard(request: Request, x_api_key: str = Header(default="")) -> None:
    """Every /ask spends LLM money, so it needs a key and a rate limit."""
    if API_KEY and not hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(401, "Missing or invalid X-API-Key.")
    if not RATE_LIMIT_PER_MIN:
        return
    # shortcut: per-process and keyed on the direct peer, so a reverse proxy makes
    # all students one client; use the gateway's limiter once there is one.
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - 60
    with _rate_lock:
        hits = _rate_hits.setdefault(ip, deque())
        while hits and hits[0] <= window_start:
            hits.popleft()
        limited = len(hits) >= RATE_LIMIT_PER_MIN
        if not limited:
            hits.append(now)
        if len(_rate_hits) > _RATE_MAX_IPS:  # IPs that went quiet would otherwise stay forever
            for key in [k for k, d in _rate_hits.items() if not d or d[-1] <= window_start]:
                del _rate_hits[key]
    if limited:
        raise HTTPException(429, "Too many requests. Please wait a minute.")


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(guard)])
def ask(req: AskRequest) -> AskResponse:
    started = time.perf_counter()
    # shortcut: "why?" means nothing without its conversation, so turns with
    # history skip the cache; cache on the condensed question if hit rate suffers.
    use_cache = req.use_cache and not req.history

    try:
        if use_cache:
            hit = CACHE.get(req.question)
            if hit is not None:
                # Kept out of _LATENCIES: near-zero hits would flatter the graph's p95.
                hit["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                return AskResponse(**hit)

        state = GRAPH.invoke(
            {"question": req.question, "history": [t.model_dump() for t in req.history]}
        )
    except Exception as exc:  # noqa: BLE001 - Qdrant or LLM down, timeout, bad response
        logger.exception("failed to answer doubt")
        raise HTTPException(503, "Could not answer right now. Please retry.") from exc

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    _LATENCIES.append(latency_ms)

    payload = {
        "answer": state.get("answer", ""),
        "route": state.get("route"),
        "sources": state.get("sources", []),
        "tool_calls": state.get("tool_calls", []),
        "cached": False,
        "cache_similarity": None,
        "latency_ms": latency_ms,
        "prompt_tokens": state.get("prompt_tokens", 0),
        "completion_tokens": state.get("completion_tokens", 0),
    }

    # Don't cache refusals or scope rejections — they'd poison later near-matches.
    # Nor numeric answers: "20 m tower" and "50 m tower" are near-identical vectors
    # with different correct answers.
    if use_cache and state.get("route") not in ("out_of_scope", "numeric"):
        if "I don't have this in my source material" not in payload["answer"]:
            CACHE.put(req.question, payload)

    return AskResponse(**payload)
