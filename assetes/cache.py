"""In-process semantic cache.

Students ask the same doubt in a hundred phrasings ("why does a ball bounce
lower", "why doesn't a ball bounce back to the same height"). An exact-match
cache never fires on those; a cosine-similarity cache does.

Scope note: this is per-process, so it is a demo-grade cache. For real
multi-replica deployment the same logic belongs behind Redis with a vector
index — the interface below is deliberately the one you'd keep.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

from app.config import (
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_SIZE,
    SEMANTIC_CACHE_THRESHOLD,
)
from app.embeddings import embed_dense_one


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class CacheEntry:
    question: str
    vector: list[float]
    payload: dict


class SemanticCache:
    def __init__(
        self,
        max_size: int = SEMANTIC_CACHE_SIZE,
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
    ):
        self.max_size = max_size
        self.threshold = threshold
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, question: str) -> dict | None:
        if not SEMANTIC_CACHE_ENABLED:
            return None
        vec = embed_dense_one(question)
        with self._lock:
            best_key, best_sim = None, 0.0
            for key, entry in self._store.items():
                sim = _cosine(vec, entry.vector)
                if sim > best_sim:
                    best_key, best_sim = key, sim
            if best_key is not None and best_sim >= self.threshold:
                self._store.move_to_end(best_key)
                self.hits += 1
                payload = dict(self._store[best_key].payload)
                payload["cached"] = True
                payload["cache_similarity"] = round(best_sim, 4)
                return payload
            self.misses += 1
        return None

    def put(self, question: str, payload: dict) -> None:
        if not SEMANTIC_CACHE_ENABLED:
            return
        vec = embed_dense_one(question)
        with self._lock:
            self._store[question] = CacheEntry(question, vec, dict(payload))
            self._store.move_to_end(question)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


CACHE = SemanticCache()
