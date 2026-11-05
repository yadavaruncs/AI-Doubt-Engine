"""Retrieval, chunking and cache behaviour, all offline."""

import pytest

from app.cache import SemanticCache
from app.ingest import chunk_text
from app.retrieval import format_context, retrieve
from tests.conftest import CORPUS


def test_chunker_respects_size_and_keeps_content():
    text = "\n\n".join(f"Paragraph {i} about physics and motion." for i in range(20))
    chunks = chunk_text(text, size=200, overlap=30)
    assert len(chunks) > 1
    assert all(chunks)
    assert "Paragraph 0" in chunks[0]


def test_chunker_hard_windows_an_oversized_paragraph():
    chunks = chunk_text("x" * 2000, size=300, overlap=0)
    assert len(chunks) >= 6
    assert max(len(c) for c in chunks) <= 300


@pytest.mark.parametrize(
    "text",
    [
        "x" * 2000,
        "\n\n".join(f"Paragraph {i} about physics and motion." for i in range(40)),
        "short intro paragraph\n\n" + "y" * 1000 + "\n\nshort outro paragraph",
    ],
)
def test_chunks_stay_within_size_including_overlap(text):
    assert max(len(c) for c in chunk_text(text, size=300, overlap=60)) <= 300


def test_oversized_paragraph_is_only_overlapped_once():
    para = "".join(f"{i:05d}" for i in range(400))
    chunks = chunk_text(para, size=300, overlap=60)
    assert sum(len(c) for c in chunks) < len(para) * 1.3
    bodies = [chunks[0]] + [c.split("\n", 1)[1] for c in chunks[1:]]
    assert "".join(bodies) == para  # nothing lost


def test_chunker_rejects_overlap_that_leaves_no_room():
    with pytest.raises(ValueError):
        chunk_text("text", size=100, overlap=99)


def test_reingest_does_not_duplicate(seeded_store):
    before = seeded_store.client().count(seeded_store.COLLECTION).count
    seeded_store.upsert(
        [{"text": t, "source": "notes.txt", "page": i} for i, t in enumerate(CORPUS, 1)]
    )
    assert seeded_store.client().count(seeded_store.COLLECTION).count == before


def test_upsert_rejects_wrong_dimension(memory_store, monkeypatch):
    monkeypatch.setattr(memory_store, "embed_dense", lambda texts: [[0.1] * 5 for _ in texts])
    with pytest.raises(ValueError, match="DENSE_DIM"):
        memory_store.upsert([{"text": "t"}])


def test_hybrid_search_returns_hits(seeded_store):
    hits = seeded_store.hybrid_search("why does a ball bounce lower", limit=4)
    assert hits
    assert all(h.source == "notes.txt" for h in hits)


def test_dense_only_baseline_arm_works(seeded_store):
    assert seeded_store.dense_only_search("Newton first law", limit=3)


@pytest.mark.parametrize("use_hybrid", [True, False])
@pytest.mark.parametrize("use_rerank", [True, False])
def test_all_four_ablation_configs_run(seeded_store, use_hybrid, use_rerank):
    """The eval harness depends on every switch combination being valid."""
    hits = retrieve(
        "coefficient of restitution",
        final_k=2,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
    )
    assert 0 < len(hits) <= 2


def test_format_context_numbers_blocks_for_citation(seeded_store):
    ctx = format_context(retrieve("Rayleigh scattering", final_k=2))
    assert ctx.startswith("[1]")
    assert "source: notes.txt" in ctx


def test_semantic_cache_hits_on_near_duplicate():
    cache = SemanticCache(max_size=8, threshold=0.80)
    cache.put("what is inertia", {"answer": "A", "route": "conceptual"})
    hit = cache.get("what is inertia")
    assert hit is not None
    assert hit["cached"] is True


def test_semantic_cache_misses_on_unrelated_question():
    cache = SemanticCache(max_size=8, threshold=0.90)
    cache.put("what is inertia", {"answer": "A"})
    assert cache.get("how do I bake sourdough bread") is None


def test_semantic_cache_evicts_lru():
    cache = SemanticCache(max_size=2, threshold=0.99)
    for i in range(5):
        cache.put(f"question number {i}", {"answer": str(i)})
    assert cache.stats()["entries"] == 2


def test_cache_hit_reports_no_tokens_spent():
    cache = SemanticCache(max_size=8, threshold=0.90)
    cache.put("what is inertia", {"answer": "A", "prompt_tokens": 500, "completion_tokens": 90})
    hit = cache.get("what is inertia")
    assert hit["prompt_tokens"] == hit["completion_tokens"] == 0


def test_cache_entries_expire():
    cache = SemanticCache(max_size=8, threshold=0.90, ttl=-1)
    cache.put("what is inertia", {"answer": "A"})
    assert cache.get("what is inertia") is None
    assert cache.stats()["entries"] == 0
