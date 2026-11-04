"""Retrieval pipeline: hybrid candidates -> cross-encoder rerank -> top-K."""

from __future__ import annotations

from app.config import CANDIDATES_K, FINAL_K
from app.embeddings import rerank
from app.store import Hit, dense_only_search, hybrid_search


def retrieve(
    query: str,
    *,
    final_k: int = FINAL_K,
    candidates_k: int = CANDIDATES_K,
    use_hybrid: bool = True,
    use_rerank: bool = True,
) -> list[Hit]:
    """The four switches here exist so the eval harness can ablate them.

    Comparing (hybrid, rerank) against (dense, no-rerank) on the same questions
    is what produces a defensible number, rather than a claim.
    """
    search = hybrid_search if use_hybrid else dense_only_search
    candidates = search(query, limit=candidates_k)
    if not candidates:
        return []

    if not use_rerank:
        return candidates[:final_k]

    scores = rerank(query, [h.text for h in candidates])
    for hit, score in zip(candidates, scores):
        hit.score = float(score)
    candidates.sort(key=lambda h: h.score, reverse=True)
    return candidates[:final_k]


def format_context(hits: list[Hit]) -> str:
    """Numbered blocks so the model can cite [1], [2] and we can check it did."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        loc = f"{h.source}" + (f", p.{h.page}" if h.page is not None else "")
        blocks.append(f"[{i}] (source: {loc})\n{h.text}")
    return "\n\n---\n\n".join(blocks)
