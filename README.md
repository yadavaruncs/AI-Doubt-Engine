# AI Doubt Engine

A retrieval-augmented physics tutor built the way a production AI system is built: hybrid retrieval, cross-encoder reranking, a tool-using agent that refuses to do its own arithmetic, and an **ablation harness that measures whether each of those choices earns its place.** One of them didn't — see [Results](#results).

Indexed corpus: [OpenStax *College Physics 2e*](https://openstax.org/details/books/college-physics-2e) (CC BY-NC-SA 4.0), 6,176 chunks. Eval: 75 scored questions plus 5 out-of-scope.

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

A **semantic cache** fronts the HTTP `/ask` endpoint: questions within a cosine threshold of a previously answered one return instantly, which is the single biggest latency and cost win in an edtech workload where thousands of students ask the same doubt in different words. Numeric answers are never cached, because "20 m tower" and "50 m tower" embed almost identically but have different answers. Entries expire after `SEMANTIC_CACHE_TTL` so a re-ingest takes effect. The evaluation harness calls the graph directly and deliberately skips the cache, so every config answers every question itself.

`POST /ask` accepts up to six prior turns in `history`. When present, a condense step first rewrites the follow-up into a standalone question, so the router, retriever and answer prompts stay conversation-free. Turns with history bypass the cache.

### Design decisions worth defending

| Decision | Why | Alternative rejected |
|---|---|---|
| RRF fusion in Qdrant | Fuses dense + sparse ranks server-side; one round trip | Fusing in Python — slower, more code |
| Cross-encoder rerank | Reads query and passage jointly. On this corpus it did *not* beat hybrid alone (see Results), so it is a switch, not a given | Trusting bi-encoder distance |
| Route *before* retrieving | Out-of-scope messages never touch the vector store | Always retrieving — wasted cost |
| SymPy via tool calling | Deterministic arithmetic, auditable steps | Letting the LLM compute |
| Semantic, not exact, cache | Students rephrase constantly | `dict[question]` — near-zero hit rate |
| Ablation harness | Turns claims into numbers | Reporting one config's score |

---

## Quick start

```bash
cp .env.example .env     # add your LLM key (or point at local Ollama) and replace API_KEY
docker compose up --build -d

# fetch the corpus the results were measured on, then index it
curl -L -o data/college-physics-2e.pdf https://assets.openstax.org/oscms-prodcms/media/documents/college-physics-2e_-_WEB.pdf
docker compose exec api python -m app.ingest --path data/ --recreate

open http://localhost:8501     # Streamlit UI
open http://localhost:8000/docs # API docs
```

Without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker run -p 127.0.0.1:6333:6333 qdrant/qdrant   # just the vector store
python -m app.ingest --path data/
ALLOW_UNAUTHENTICATED=1 uvicorn app.main:app --reload   # allows an empty API_KEY, local use only
```

**Replace the example key.** `API_KEY` in `.env.example` is public, so the API refuses to start with it. Generate your own with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. It also refuses an empty key unless `ALLOW_UNAUTHENTICATED=1` is set, which is for local use only. Clients send the key in the `X-API-Key` header.

Behind a reverse proxy, set `FORWARDED_ALLOW_IPS` to the proxy's IP, otherwise all clients share one rate limit. The limit is per process, so running several replicas multiplies it.

Under Compose, Qdrant has no host port: only the other containers can reach it, so run ingest with `docker compose exec` as above.

Point IDs are derived from chunk content, so re-running ingest overwrites instead of duplicating. Use `--recreate` whenever a source file is edited or removed, otherwise its old chunks stay.

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

# RAGAS on one config only (each config costs one answer run per question plus judge calls)
python -m eval.run_eval --dataset eval/dataset.jsonl --ragas-configs hybrid_rerank
```

It runs the same questions through four configurations and writes [eval/results.csv](eval/results.csv), plus [eval/per_question.csv](eval/per_question.csv) with the rank of the first relevant chunk for every question and config.

A retrieved chunk counts as relevant if it contains one of the question's `must_contain` phrases (case, whitespace and apostrophe style normalised). Every phrase was checked against the book text before the run, and the dataset was frozen before any config was scored. Questions are tagged `term` (uses the textbook's wording, 43) or `paraphrase` (a student's wording that avoids the term, 32).

### Results

Measured 2026-11-13 on CPU: fastembed `bge-small-en-v1.5` + `Qdrant/bm25`, reranker `ms-marco-MiniLM-L-6-v2`, 20 candidates → top 4.

| config | hit_rate@4 | MRR | hit_rate@4 (paraphrase) | MRR (paraphrase) | p95 retrieval |
|---|---|---|---|---|---|
| dense_only | 0.920 | 0.798 | 0.844 | 0.625 | 2.16 s |
| dense_rerank | 0.907 | 0.809 | 0.844 | 0.661 | 2.98 s |
| **hybrid_only** | **0.947** | **0.859** | **0.938** | 0.732 | 2.15 s |
| hybrid_rerank | 0.947 | 0.838 | 0.938 | **0.734** | 3.05 s |

What it showed:

- **Hybrid search is the win, and it comes from paraphrased questions.** On textbook-term questions dense retrieval already hits 97.7%, so there is little to gain. On paraphrases, adding BM25 lifts hit rate from 84% to 94% and MRR from 0.63 to 0.73. For example, "Why is the sky blue?" moves from rank 4 to rank 1 and "Why can some insects walk on water?" from rank 2 to rank 1. Paraphrases still share some content words with the right passage, and BM25 rewards those words where the small dense model drifts to related topics.
- **The cross-encoder did not earn its latency here.** On top of hybrid it improved 7 questions and worsened 11, MRR fell from 0.859 to 0.838, and it added about 0.9 s at p95 on CPU. A likely cause, not yet verified: the reranker was trained on web search queries and favours the book's end-of-chapter exercises, which echo a student's wording, over the explanatory passage. The next experiment is to tag chunks by section type and rerank only explanatory text, or to try a larger reranker.
- **Numeric questions are the weak spot.** Three of hybrid_rerank's four misses are numeric ("dropped from a 20 m tower", "2 kg at 10 m/s", "heat 1 kg of water"). The numbers and setup match worked examples about *other* values, not the section that states the formula. The kinetic-energy question was missed by every config.
- **These are directions, not proof.** Hybrid vs dense: 14 questions better, 7 worse (two-sided sign test p ≈ 0.19). Hybrid_rerank vs hybrid: 7 better, 11 worse (p ≈ 0.48). With 75 questions, a difference of a few questions is within noise.

### Answer quality (RAGAS)

Measured 2026-11-14, fully offline on a laptop RTX 4060: answers by `qwen2.5:7b`, judged by `llama3` 8B, both on Ollama. The judge is a different model from the answerer, so the scores are not self-graded. Only `hybrid_rerank` was scored, on every third question of the dataset (27). The 2 out-of-scope probes among them were correctly refused and are excluded, leaving 25. Raw output: [eval/ragas_results.csv](eval/ragas_results.csv).

| metric | score | answers scored |
|---|---|---|
| faithfulness | 0.780 | 14 of 25 |
| answer_relevancy | 0.710 | 25 of 25 |
| context_precision | 0.929 | 25 of 25 |

- **Faithfulness is the number to watch.** On average 78% of the claims in an answer are supported by the retrieved passages, so about one claim in five comes from the model rather than the book. It rests on 14 answers: the 8B judge returned unparseable output for the other 11, and RAGAS drops those.
- **Retrieved context is mostly on target.** Context precision of 0.93 agrees with the retrieval table: the passages the judge found relevant were ranked above the ones it did not.
- **This is not the deployed model.** The app is configured for `qwen/qwen3.8-27b` on Groq. A clean run needs about 120,000 tokens (about 1,500 per answer), within the free tier's 200,000 a day, but earlier failed runs had used up the day's quota, so this run was done offline. A 7B answerer and an 8B judge likely understate quality and add noise. One numeric question (thin-lens image distance) ran out of tool rounds and replied "I couldn't finish this calculation reliably"; the kinetic-energy question used the calculator and got 100 J.

Caveats: the index stops at page 1479 of 1671, because ingest was cut short. That drops the second half of chapter 32 (nuclear fission and fusion), chapters 33–34 and the appendices, though every scored question still has a matching passage in the index. Latency was measured while the API was also serving requests, so read it as relative, not absolute. Hybrid retrieval is not fully deterministic: four runs of the same code on the same index gave hybrid_only MRR of 0.834, 0.844, 0.852 and 0.859, while dense results were identical every time. Read the hybrid MRR figures as carrying about ±0.01 of run-to-run noise; the cause is not yet pinned down. By default the RAGAS judge is the same model that wrote the answers (`JUDGE_MODEL`, `JUDGE_BASE_URL` and `JUDGE_API_KEY` fall back to the `LLM_*` values), which makes faithfulness self-evaluated and likely inflated. Set the `JUDGE_*` variables to a different model for a fair score, as the run above did. Out-of-scope rejections are excluded from the RAGAS scores, since they have no retrieved context to judge.

Run on a different corpus or change a setting (`DENSE_MODEL`, `CHUNK_SIZE`, `CANDIDATES_K`) and re-measure. `per_question.csv` shows which questions moved.

---

## Tests

```bash
pip install pytest httpx && python -m pytest tests/ -q
```

No network, no API key, no Qdrant server — embeddings are stubbed and Qdrant runs in-memory. Includes a regression suite for a real vulnerability: SymPy's `parse_expr` calls `eval()` underneath, and without the guard in `app/tools.py` an expression like `__import__('os').system(...)` executes. Never pass model-generated strings to a math parser without a allowlist.

---

## Project layout

```
app/
  config.py      env-driven settings
  embeddings.py  lazily-loaded dense / sparse / cross-encoder models
  store.py       Qdrant collection, hybrid RRF search, dense-only baseline
  retrieval.py   candidates → rerank → top-k, with ablation switches
  tools.py       SymPy calculator + solver, hardened against injection
  graph.py       LangGraph agent: condense → route → retrieve → answer (+ tool loop)
  cache.py       semantic cache
  ingest.py      PDF/text → paragraph-aware chunks → Qdrant
  main.py        FastAPI: /ask /healthz /metrics
eval/
  dataset.jsonl  80 golden questions with must_contain keys and term/paraphrase tags
  run_eval.py    four-way ablation harness
  results.csv    the numbers in the Results table
  per_question.csv  first-relevant rank per question and config
tests/           offline test suite
ui/              Streamlit front end showing sources, route, latency
```

---

## Next steps

- Run RAGAS (faithfulness, answer relevancy, context precision) with a paid key
- Rerank only explanatory sections, or try a larger cross-encoder, and re-measure against `hybrid_only`
- Expand the numeric questions, the weakest category
- Handwritten-answer grading with rubric-based scoring and agreement (Cohen's κ) against human scores
- Multimodal doubts: a student photographs a problem and the system reads the diagram
- Redis-backed shared semantic cache for multi-replica deployment
- Streaming responses via SSE to cut perceived latency

---

## Credits

The question-type routing idea was informed by [tan7vir/Physics-Chatbot-using-Advanced-RAG](https://github.com/tan7vir/Physics-Chatbot-using-Advanced-RAG). The implementation here is independent, built on a current stack with hybrid retrieval, reranking, evaluation and tests added.

## License

MIT
