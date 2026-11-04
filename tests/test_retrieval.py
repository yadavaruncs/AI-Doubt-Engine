"""Retrieval, chunking and cache behaviour, all offline."""

import pytest

from app.cache import SemanticCache
from app.ingest import chunk_text
from app.retrieval import format_context, retrieve


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
