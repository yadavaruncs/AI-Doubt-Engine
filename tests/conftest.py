"""Stub the embedding models and point Qdrant at :memory:.

This lets the retrieval, cache and graph tests run in CI with no network, no
ONNX download, no Qdrant server and no LLM key.
"""

import hashlib

import numpy as np
import pytest

DIM = 384


def _vec(text: str) -> list[float]:
    base = np.zeros(DIM)
    for tok in set(text.lower().split()):
        seed = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "little")
        base += np.random.default_rng(seed).normal(size=DIM)
    if not np.any(base):
        base = np.ones(DIM)
    return (base / np.linalg.norm(base)).tolist()


class _SparseVec:
    def __init__(self, text: str):
        toks = sorted({t.strip(".,?!") for t in text.lower().split() if t.strip(".,?!")})
        self.indices = np.array(
            [
                int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little") % 100_000
                for t in toks
            ],
            dtype=np.int64,
        )
        self.values = np.ones(len(toks), dtype=np.float32)


@pytest.fixture(autouse=True)
def stub_models(monkeypatch):
    import app.cache as cache_mod
    import app.embeddings as emb
    import app.retrieval as retrieval
    import app.store as store

    def embed_dense(texts):
        return [_vec(t) for t in texts]

    def embed_sparse(texts):
        return [_SparseVec(t) for t in texts]

    def rerank(query, docs):
        q = set(query.lower().split())
        return [float(len(q & set(d.lower().split()))) for d in docs]

    for mod in (emb, store):
        monkeypatch.setattr(mod, "embed_dense", embed_dense, raising=False)
        monkeypatch.setattr(mod, "embed_sparse", embed_sparse, raising=False)
    monkeypatch.setattr(emb, "embed_dense_one", lambda t: _vec(t), raising=False)
    monkeypatch.setattr(cache_mod, "embed_dense_one", lambda t: _vec(t), raising=False)
    monkeypatch.setattr(retrieval, "rerank", rerank, raising=False)


@pytest.fixture
def memory_store(monkeypatch):
    from qdrant_client import QdrantClient

    import app.store as store

    monkeypatch.setattr(store, "_client", QdrantClient(":memory:"))
    store.ensure_collection(recreate=True)
    return store


CORPUS = [
    "Newton's first law states that a body stays at rest or in uniform motion "
    "unless acted on by a net external force. This property is called inertia.",
    "A bouncing ball loses kinetic energy to heat and sound in each inelastic "
    "collision, so it rebounds lower. The coefficient of restitution is the "
    "ratio of separation speed to approach speed.",
    "The equations of motion for uniform acceleration are v = u + at and "
    "v squared equals u squared plus two a s.",
    "The sky appears blue because of Rayleigh scattering, which affects shorter "
    "wavelengths far more strongly than longer wavelengths.",
]


@pytest.fixture
def seeded_store(memory_store):
    memory_store.upsert(
        [{"text": t, "source": "notes.txt", "page": i} for i, t in enumerate(CORPUS, 1)]
    )
    return memory_store
