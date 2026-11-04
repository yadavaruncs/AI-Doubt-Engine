"""FastAPI service.

POST /ask        -> answer a doubt
GET  /healthz    -> liveness + collection status
GET  /metrics    -> cache stats and rolling latency percentiles
"""

from __future__ import annotations

import time
from collections import deque
from statistics import median

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.cache import CACHE
from app.config import COLLECTION
from app.graph import GRAPH
from app.store import client, ensure_collection

app = FastAPI(title="AI Doubt Engine", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_LATENCIES: deque[float] = deque(maxlen=500)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
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


@app.on_event("startup")
def _startup() -> None:
    try:
        ensure_collection()
    except Exception as exc:  # noqa: BLE001 - don't crash-loop if Qdrant is slow to boot
        print(f"[startup] Qdrant not ready yet: {exc}")


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


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    started = time.perf_counter()

    if req.use_cache:
        hit = CACHE.get(req.question)
        if hit is not None:
            hit["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            _LATENCIES.append(hit["latency_ms"])
            return AskResponse(**hit)

    state = GRAPH.invoke({"question": req.question})
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
    if req.use_cache and state.get("route") != "out_of_scope":
        if "I don't have this in my source material" not in payload["answer"]:
            CACHE.put(req.question, payload)

    return AskResponse(**payload)
