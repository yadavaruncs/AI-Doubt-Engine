"""Qdrant collection management and hybrid (dense + sparse) search."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.config import (
    CANDIDATES_K,
    COLLECTION,
    DENSE_DIM,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from app.embeddings import embed_dense, embed_sparse

DENSE_VEC = "dense"
SPARSE_VEC = "sparse"


@dataclass
class Hit:
    text: str
    source: str
    page: int | None
    score: float


_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return _client


def ensure_collection(recreate: bool = False) -> None:
    c = client()
    exists = c.collection_exists(COLLECTION)
    if exists and recreate:
        c.delete_collection(COLLECTION)
        exists = False
    if exists:
        return
    c.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE_VEC: models.VectorParams(
                size=DENSE_DIM, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VEC: models.SparseVectorParams(
                modifier=models.Modifier.IDF  # required for BM25-style scoring
            )
        },
    )


def upsert(chunks: list[dict], batch_size: int = 32) -> int:
    """chunks: [{"text": str, "source": str, "page": int|None}, ...]"""
    ensure_collection()
    c = client()
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [b["text"] for b in batch]
        dense = embed_dense(texts)
        sparse = embed_sparse(texts)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VEC: dense[j],
                    SPARSE_VEC: models.SparseVector(
                        indices=sparse[j].indices.tolist(),
                        values=sparse[j].values.tolist(),
                    ),
                },
                payload={
                    "text": batch[j]["text"],
                    "source": batch[j].get("source", "unknown"),
                    "page": batch[j].get("page"),
                },
            )
            for j in range(len(batch))
        ]
        c.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
    return total


def hybrid_search(query: str, limit: int = CANDIDATES_K) -> list[Hit]:
    """Dense + BM25 candidates fused server-side with Reciprocal Rank Fusion.

    RRF is what makes this *hybrid* rather than two separate searches: a chunk
    that ranks well on either arm survives, which matters for physics, where
    'coefficient of restitution' needs lexical match but 'why does a ball bounce
    lower each time' needs semantic match.
    """
    c = client()
    dense_q = embed_dense([query])[0]
    sparse_q = embed_sparse([query])[0]

    result = c.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_q, using=DENSE_VEC, limit=limit),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_q.indices.tolist(),
                    values=sparse_q.values.tolist(),
                ),
                using=SPARSE_VEC,
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [
        Hit(
            text=p.payload.get("text", ""),
            source=p.payload.get("source", "unknown"),
            page=p.payload.get("page"),
            score=float(p.score),
        )
        for p in result.points
    ]


def dense_only_search(query: str, limit: int = CANDIDATES_K) -> list[Hit]:
    """Baseline arm — used by the eval harness to prove hybrid actually helps."""
    c = client()
    result = c.query_points(
        collection_name=COLLECTION,
        query=embed_dense([query])[0],
        using=DENSE_VEC,
        limit=limit,
        with_payload=True,
    )
    return [
        Hit(
            text=p.payload.get("text", ""),
            source=p.payload.get("source", "unknown"),
            page=p.payload.get("page"),
            score=float(p.score),
        )
        for p in result.points
    ]
