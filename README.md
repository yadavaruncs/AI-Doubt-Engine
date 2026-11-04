# AI Doubt Engine

A retrieval-augmented physics tutor built the way a production AI system is built: hybrid retrieval, cross-encoder reranking, a tool-using agent that refuses to do its own arithmetic, and an **ablation harness that proves each of those choices earns its place.**

> **Status: scaffold.** The pipeline runs end to end. The numbers in the results table below are placeholders until you run the eval against your own corpus. Replace them before putting this on a resume — see [Make it yours](#make-it-yours).

---

## Why this exists

Most "chat with your PDF" projects are a single `similarity_search(k=3)` call wrapped in a prompt. They demo well and fall over on real questions, because:

- **Dense-only retrieval misses exact terms.** A student searching "coefficient of restitution" needs lexical matching; one asking "why doesn't a ball bounce back as high" needs semantic matching. You need both.
- **Top-k by vector distance is not relevance.** A bi-encoder embeds the query and document separately; a cross-encoder reads them together and is substantially more accurate at ranking. Retrieve 20 cheaply, rerank to 4 precisely.
- **Language models are unreliable at arithmetic.** Any tutor that lets the model compute `sqrt(2 × 9.8 × 20)` in its head will eventually tell a student the wrong answer with total confidence.
- **Nobody measures anything.** Claims like "improved accuracy" with no baseline are worthless in an interview.

This project addresses each of those explicitly and measures the result.

---

## Architecture

```
                 ┌────────────┐
   question ────>│   router   │  classifies: conceptual | numeric | out_of_scope
                 └─────┬──────┘  (cheap model, 8 output tokens)
                       │
          ┌────────────┴────────────┐
          │                         │
   out_of_scope                  retriever
          │                         │
          │            ┌────────────┴────────────┐
          │            │   hybrid search         │  dense (BGE) + BM25 sparse
          │            │   → RRF fusion          │  fused server-side by Qdrant
          │            │   → cross-encoder       │  20 candidates → top 4
          │            └────────────┬────────────┘
          │                         │
          │            ┌────────────┴────────────┐
          │            │                         │
          │      answer_conceptual         answer_numeric
          │            │                         │
          │            │                   ┌─────┴─────┐
          │            │                   │  SymPy    │ tool loop (bounded, 4 rounds)
          │            │                   └─────┬─────┘
          └────────────┴─────────────┬───────────┘
                                     ▼
                              answer + citations
```

A **semantic cache** sits in front of the whole graph: questions within a cosine threshold of a previously answered one return instantly, which is the single biggest latency and cost win in an edtech workload where thousands of students ask the same doubt in different words.

### Design decisions worth defending

| Decision | Why | Alternative rejected |
|---|---|---|
| RRF fusion in Qdrant | Fuses dense + sparse ranks server-side; one round trip | Fusing in Python — slower, more code |
| Cross-encoder rerank | Reads query and passage jointly; large precision gain | Trusting bi-encoder distance |
| Route *before* retrieving | Out-of-scope messages never touch the vector store | Always retrieving — wasted cost |
| SymPy via tool calling | Deterministic arithmetic, auditable steps | Letting the LLM compute |
| Semantic, not exact, cache | Students rephrase constantly | `dict[question]` — near-zero hit rate |
| Ablation harness | Turns claims into numbers | Reporting one config's score |

---

## Quick start

```bash
cp .env.example .env     # add your LLM key, or point at local Ollama
docker compose up --build -d

# add your source material, then index it
cp /path/to/physics.pdf data/
docker compose exec api python -m app.ingest --path data/ --recreate

open http://localhost:8501     # Streamlit UI
open http://localhost:8000/docs # API docs
```

Without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker run -p 6333:6333 qdrant/qdrant   # just the vector store
python -m app.ingest --path data/
uvicorn app.main:app --reload
```

**No API key?** Set `LLM_BASE_URL=http://localhost:11434/v1` and run Ollama locally. Embeddings and reranking run on CPU via fastembed and never need a key.

---

## Running the evaluation

This is the part that matters.

```bash
# retrieval metrics only — no LLM calls, no cost
python -m eval.run_eval --dataset eval/dataset.jsonl --no-ragas

# full run, adds RAGAS faithfulness / relevancy / context precision
pip install -r requirements-eval.txt
python -m eval.run_eval --dataset eval/dataset.jsonl
```

It runs the same questions through four configurations and writes `eval/results.csv`:

| config | hit_rate@4 | mrr | faithfulness | p95 retrieval |
|---|---|---|---|---|
| dense_only | — | — | — | — |
| dense_rerank | — | — | — | — |
| hybrid_only | — | — | — | — |
| hybrid_rerank | — | — | — | — |

**The delta between `dense_only` and `hybrid_rerank` is your headline number.** Fill this table in from your own run. An interviewer will ask you to explain it, so make sure you can — including the cases where reranking *didn't* help.

`eval/dataset.jsonl` ships with 10 starter questions. Expand it to 50+ against your corpus; below ~30 the metrics are noise.

---

## Tests

```bash
pip install pytest && python -m pytest tests/ -q
```

30 tests, no network, no API key, no Qdrant server — embeddings are stubbed and Qdrant runs in-memory. Includes a regression suite for a real vulnerability: SymPy's `parse_expr` calls `eval()` underneath, and without the guard in `app/tools.py` an expression like `__import__('os').system(...)` executes. Never pass model-generated strings to a math parser without a allowlist.

---

## Project layout

```
app/
  config.py      env-driven settings
  embeddings.py  lazily-loaded dense / sparse / cross-encoder models
  store.py       Qdrant collection, hybrid RRF search, dense-only baseline
  retrieval.py   candidates → rerank → top-k, with ablation switches
  tools.py       SymPy calculator + solver, hardened against injection
  graph.py       LangGraph agent: route → retrieve → answer (+ tool loop)
  cache.py       semantic cache
  ingest.py      PDF/text → paragraph-aware chunks → Qdrant
  main.py        FastAPI: /ask /healthz /metrics
eval/
  dataset.jsonl  golden questions with must_contain keys
  run_eval.py    four-way ablation harness
tests/           offline test suite
ui/              Streamlit front end showing sources, route, latency
```

---

## Make it yours

This scaffold is a starting point, not a finished portfolio piece. To make it genuinely yours:

1. **Use a corpus you care about.** NCERT, your own class notes, a specific JEE topic. The source material is what makes the project specific rather than generic.
2. **Expand the eval set to 50+ questions** with real `must_contain` keys, and run it. Record the numbers.
3. **Then change something and re-measure.** Try a different embedding model, chunk size, or `CANDIDATES_K`. The story "I tried X, it helped by N%, here's why" is worth more than any feature you add.
4. **Write up what didn't work.** The honest failure analysis is the most convincing section of any project README.
5. Replace the placeholder table above with your real results and delete this section.

### Worthwhile extensions
- Handwritten-answer grading with rubric-based scoring and agreement (Cohen's κ) against human scores
- Multimodal doubts: a student photographs a problem and the system reads the diagram
- Redis-backed shared semantic cache for multi-replica deployment
- Streaming responses via SSE to cut perceived latency

---

## Credits

The multi-query expansion and question-type routing ideas were informed by [tan7vir/Physics-Chatbot-using-Advanced-RAG](https://github.com/tan7vir/Physics-Chatbot-using-Advanced-RAG). The implementation here is independent, built on a current stack with hybrid retrieval, reranking, evaluation and tests added.

## License

MIT
