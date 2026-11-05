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
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from app.config import (
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_SIZE,
    SEMANTIC_CACHE_THRESHOLD,
    SEMANTIC_CACHE_TTL,
)
from app.embeddings import embed_dense_one


def _unit(vector: list[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm else v


@dataclass
class CacheEntry:
    question: str
    vector: np.ndarray  # unit length, so a dot product is the cosine
    payload: dict
    expires_at: float


class SemanticCache:
    def __init__(
        self,
        max_size: int = SEMANTIC_CACHE_SIZE,
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
        ttl: float = SEMANTIC_CACHE_TTL,
    ):
        self.max_size = max_size
        self.threshold = threshold
        # shortcut: ingest runs in another process and can't clear this, so entries
        # go stale for up to ttl after a re-ingest; move to Redis to invalidate.
        self.ttl = ttl
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        # (keys, entries, matrix) in matching LRU order, one unit vector per row.
        # Rebuilt after a change, never edited in place, so get() can use it after
        # dropping the lock.
        self._snapshot: tuple[list[str], list[CacheEntry], np.ndarray] | None = None
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, question: str) -> dict | None:
        if not SEMANTIC_CACHE_ENABLED:
            return None
        vec = _unit(embed_dense_one(question))
        now = time.monotonic()
        with self._lock:
            expired = [k for k, e in self._store.items() if e.expires_at <= now]
            for key in expired:
                del self._store[key]
            if expired:
                self._snapshot = None
            if self._snapshot is None and self._store:
                keys, entries = list(self._store), list(self._store.values())
                self._snapshot = (keys, entries, np.stack([e.vector for e in entries]))
            snapshot = self._snapshot

        best_key, best_entry, best_sim = None, None, 0.0
        if snapshot is not None:
            keys, entries, matrix = snapshot
            sims = matrix @ vec
            # BLAS can split identical rows by one ulp; rounding keeps ties in LRU order.
            best = int(np.argmax(np.round(sims, 12)))
            if sims[best] > 0:  # nothing unrelated matches, even with threshold <= 0
                best_key, best_entry, best_sim = keys[best], entries[best], float(sims[best])

        with self._lock:
            if best_entry is not None and best_sim >= self.threshold:
                if best_key in self._store:  # may have been evicted meanwhile
                    self._store.move_to_end(best_key)
                    self._snapshot = None  # ties go to the least recently used entry
                self.hits += 1
                payload = dict(best_entry.payload)
                payload["cached"] = True
                payload["cache_similarity"] = round(best_sim, 4)
                payload["prompt_tokens"] = payload["completion_tokens"] = 0  # none spent
                return payload
            self.misses += 1
        return None

    def put(self, question: str, payload: dict) -> None:
        if not SEMANTIC_CACHE_ENABLED:
            return
        vec = _unit(embed_dense_one(question))
        with self._lock:
            self._store[question] = CacheEntry(
                question, vec, dict(payload), time.monotonic() + self.ttl
            )
            self._snapshot = None
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
