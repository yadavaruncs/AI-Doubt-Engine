"""Ablation harness: measures whether hybrid search and reranking actually help.

This is the file that turns "I built a RAG app" into a number you can defend in
an interview. It runs the SAME questions through four configurations and reports
retrieval quality plus RAGAS answer-quality metrics side by side.

    python -m eval.run_eval --dataset eval/dataset.jsonl
    python -m eval.run_eval --dataset eval/dataset.jsonl --no-ragas   # retrieval only, no API cost

Output: eval/results.csv and a printed table.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from app.config import FINAL_K, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from app.embeddings import embed_dense, embed_dense_one
from app.retrieval import retrieve

CONFIGS = {
    "dense_only":      dict(use_hybrid=False, use_rerank=False),
    "dense_rerank":    dict(use_hybrid=False, use_rerank=True),
    "hybrid_only":     dict(use_hybrid=True,  use_rerank=False),
    "hybrid_rerank":   dict(use_hybrid=True,  use_rerank=True),
}


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _contains(hit_text: str, needles: list[str]) -> bool:
    low = hit_text.lower()
    return any(n.lower() in low for n in needles)


def retrieval_scores(rows: list[dict], cfg: dict) -> dict:
    """Hit-rate@k and MRR against the `must_contain` keys in the dataset."""
    hits_at_k, reciprocal_ranks, latencies = [], [], []

    for row in rows:
        needles = row.get("must_contain") or []
        if not needles:
            continue
        started = time.perf_counter()
        results = retrieve(row["question"], final_k=FINAL_K, **cfg)
        latencies.append((time.perf_counter() - started) * 1000)

        rank = next(
            (i for i, h in enumerate(results, start=1) if _contains(h.text, needles)),
            None,
        )
        hits_at_k.append(1 if rank else 0)
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    n = len(hits_at_k) or 1
    latencies.sort()
    return {
        "n": len(hits_at_k),
        f"hit_rate@{FINAL_K}": round(sum(hits_at_k) / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "retrieval_p95_ms": round(
            latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)], 1
        )
        if latencies
        else 0.0,
    }


def ragas_scores(rows: list[dict], cfg: dict) -> dict:
    """Faithfulness / answer-relevancy / context-precision via RAGAS.

    Requires an LLM key (RAGAS uses a judge model). Skip with --no-ragas.
    """
    from datasets import Dataset
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        faithfulness,
    )

    from app.graph import GRAPH

    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for row in rows:
        # The contexts must be the ones this answer was generated from.
        state = GRAPH.invoke(
            {"question": row["question"], "retrieval": {"final_k": FINAL_K, **cfg}}
        )
        records["question"].append(row["question"])
        records["answer"].append(state.get("answer", ""))
        records["contexts"].append([h.text for h in state.get("hits", [])] or [""])
        records["ground_truth"].append(row.get("ground_truth", ""))

    class LocalEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return embed_dense(texts)

        def embed_query(self, text):
            return embed_dense_one(text)

    metrics = [faithfulness, answer_relevancy, context_precision]
    # RAGAS defaults to OpenAI; use the configured endpoint so Groq/Ollama work too.
    result = evaluate(
        Dataset.from_dict(records),
        metrics=metrics,
        llm=ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY or "not-needed-for-local",
        ),
        embeddings=LocalEmbeddings(),
    )
    scores = result.to_pandas()
    return {m.name: round(float(scores[m.name].mean()), 4) for m in metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/dataset.jsonl")
    ap.add_argument("--no-ragas", action="store_true", help="retrieval metrics only")
    ap.add_argument("--out", default="eval/results.csv")
    args = ap.parse_args()

    rows = load_dataset(Path(args.dataset))
    print(f"Loaded {len(rows)} eval questions.\n")

    table = []
    for name, cfg in CONFIGS.items():
        print(f"--- {name} ---")
        scores = {"config": name, **retrieval_scores(rows, cfg)}
        if not args.no_ragas:
            scores.update(ragas_scores(rows, cfg))
        table.append(scores)
        print("  " + "  ".join(f"{k}={v}" for k, v in scores.items() if k != "config"))
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in table for k in r}, key=lambda k: (k != "config", k))
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    print(f"Wrote {out}")
    print(
        "\nPut the hybrid_rerank vs dense_only delta in your README and your resume.\n"
        "That delta is the number an interviewer will ask you to explain."
    )


if __name__ == "__main__":
    main()
