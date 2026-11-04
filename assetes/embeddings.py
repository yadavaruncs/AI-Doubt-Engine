"""Lazily-loaded embedding / reranking models.

Loaded once per process and reused. fastembed downloads ONNX weights on first
use and runs them on CPU, so none of this needs an API key or a GPU.
"""

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import DENSE_MODEL, RERANK_MODEL, SPARSE_MODEL


@lru_cache(maxsize=1)
def dense_encoder() -> TextEmbedding:
    return TextEmbedding(model_name=DENSE_MODEL)


@lru_cache(maxsize=1)
def sparse_encoder() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL)


@lru_cache(maxsize=1)
def reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANK_MODEL)


def embed_dense(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in dense_encoder().embed(texts)]


def embed_dense_one(text: str) -> list[float]:
    return embed_dense([text])[0]


def embed_sparse(texts: list[str]):
    """Yields objects with .indices and .values (Qdrant sparse vector form)."""
    return list(sparse_encoder().embed(texts))


def rerank(query: str, documents: list[str]) -> list[float]:
    """Cross-encoder relevance scores, one per document. Higher is better."""
    if not documents:
        return []
    return list(reranker().rerank(query, documents))
