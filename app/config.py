"""Central configuration. Everything is env-overridable so the same image
runs locally against Ollama and in the cloud against a hosted model."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------- LLM
# OpenAI-compatible endpoint. Works with OpenAI, Groq, Together, or a local
# Ollama server (http://localhost:11434/v1, api_key can be anything).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", LLM_MODEL)
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
# RAGAS judge. Defaults to the answering model, which means self-evaluation;
# point these at a different model for an unbiased faithfulness score.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", LLM_BASE_URL)
# Never send the answering model's key to a different endpoint.
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", LLM_API_KEY if JUDGE_BASE_URL == LLM_BASE_URL else "")

TOOL_TIMEOUT = float(os.getenv("TOOL_TIMEOUT", "5"))  # seconds per SymPy call
TOOL_MEMORY_MB = int(os.getenv("TOOL_MEMORY_MB", "1024"))  # enforced on Linux only

# ---------------------------------------------------------------- Vector store
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = os.getenv("QDRANT_COLLECTION", "physics_doubts")

# ---------------------------------------------------------------- Models
# fastembed runs these on CPU with no API key, so ingestion works offline.
DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
DENSE_DIM = int(os.getenv("DENSE_DIM", "384"))  # bge-small-en-v1.5

# ---------------------------------------------------------------- Retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
CANDIDATES_K = int(os.getenv("CANDIDATES_K", "20"))  # pre-rerank
FINAL_K = int(os.getenv("FINAL_K", "4"))  # post-rerank, into the prompt

# ---------------------------------------------------------------- Cache
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "1") == "1"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.93"))
SEMANTIC_CACHE_SIZE = int(os.getenv("SEMANTIC_CACHE_SIZE", "512"))
SEMANTIC_CACHE_TTL = float(os.getenv("SEMANTIC_CACHE_TTL", "3600"))  # seconds

# ---------------------------------------------------------------- API
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
API_KEY = os.getenv("API_KEY", "")  # required unless ALLOW_UNAUTHENTICATED=1
EXAMPLE_API_KEY = "i7R48FI6S1tD5DUcVQOxTcebStkrO-FMjkd9qzR5VYU"  # public, in .env.example
ALLOW_UNAUTHENTICATED = os.getenv("ALLOW_UNAUTHENTICATED", "") == "1"
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))  # per client IP, 0 = off
