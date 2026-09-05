# RGUKT Academic RAG Chatbot

A grounded **Retrieval-Augmented Generation** assistant for RGUKT academic regulations. It answers **only from indexed PDFs**, with **page-level citations**, a **confidence gate**, and **answer verification**.

**Status:** RAG core and verifier are **frozen**. Quality is measured against a **19-question benchmark** (`data/baseline_final.json`). Prefer UX, packaging, and deployment over further retrieval tuning.

---

## Architecture

Embeddings and search are **local**. Groq is used only as the **chat LLM** (draft generation, verification, and occasional follow-up query rewrite)—**not** as the embedding model.

```text
PDF
 ↓  ingest.py          structure-aware extract (pages, section headings)
 ↓  embed.py           pack blocks → paraphrase-MiniLM-L3-v2 vectors
 ↓  vector_store.py    FAISS IndexFlatL2
 ↓
User question
 ↓  chat.py            optional Groq rewrite (short / follow-up only)
 ↓  retrieve.py        local intent expand (requirement vs consequence)
 ↓                     FAISS top-30  +  BM25 top-30
 ↓                     RRF merge → keep 25
 ↓                     CrossEncoder ms-marco-MiniLM-L-2-v2
 ↓                     lexical / number / section boost
 ↓                     ≤1 chunk per (document, page)
 ↓  chat.py            confidence gate → abstain or draft via Groq
 ↓  verify.py          PASS / REVISE / FAIL (enabled by default)
 ↓  api_server.py      FastAPI JSON
 ↓  frontend/          React + Vite UI (citations, chat history)
```

| Component | Model / tech | Role |
| --- | --- | --- |
| Embedding | `paraphrase-MiniLM-L3-v2` | Document & query vectors |
| Dense index | FAISS (`faiss_index.index`) | Semantic nearest neighbors |
| Sparse | BM25 (`bm25.py`) | Exact terms / numbers |
| Fusion | Reciprocal Rank Fusion | Merge dense + sparse |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-2-v2` | Pairwise relevance |
| Generation / verify | Groq chat (`GROQ_MODEL`) | Draft + claim check |

---

## Frozen quality baseline

Source: `data/baseline_final.json` (19 cases, page/section/keyword evidence; chunk IDs ignored).

| Metric | Value |
| --- | ---: |
| Recall@1 / @3 / @5 | **94.7%** / **100%** / **100%** |
| MRR / nDCG@5 | **96.5%** / **96.8%** |
| Faithfulness / Correctness | **82.1%** / **85.3%** |
| Citation valid ID / grounded | **100%** / **~89%** |
| OOD refusal / false refusal | **100%** / **0%** |
| Avg latency | **~20.6 s** (verify ~60%, generation ~37%, retrieval &lt;3%) |

Precision@5 (~44%) is left alone on purpose: the product returns a small `top_k` and ranks the right page first.

Latency detail: `data/LATENCY_PROFILE.md`. Smoke results: `data/SMOKE_TEST.md`.

---

## Quick start

### Prerequisites

- Python **3.10+**
- Node.js **18+** (for the UI)
- [Groq API key](https://console.groq.com/keys)

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# Set GROQ_API_KEY=... and a model id that exists on your Groq account
```

**Important:** `GROQ_MODEL` must be a model **available on your account**. Defaults in code may 404; this project was smoke-tested with `qwen/qwen3.8-27b`. Empty or hanging responses from some `gpt-oss` ids are **config failures**, not RAG bugs.

### Index (once, or after PDF / chunking changes)

Put PDFs under `data/` (default handbook path is used by `embed.py`).

```powershell
python embed.py
python vector_store.py
```

Creates `embeddings.pkl` and `faiss_index.index`. Do **not** re-index casually—refresh `baseline_final.json` after intentional re-ingest.

### Run API + UI

**Terminal A (project root, venv active):**

```powershell
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

**Terminal B:**

```powershell
cd frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173** (chat UI). Port **8000** is the API only (`GET /` explains that). Vite proxies `/api` → `:8000`.

CLI chat (optional): `python chat.py`

---

## Evaluation & smoke tests

```powershell
# Full 19-question suite (uses Groq quota; slow)
python evaluate_rag.py

# Live product smoke against running API
python scripts/final_smoke.py

# Offline / API hardening checks
python scripts/hardening_smoke.py --offline
python scripts/hardening_smoke.py

# Retrieval debug for one query
python diagnose_retrieval.py "What is the minimum attendance requirement?"
```

Regression to keep green: *“What is the minimum attendance requirement?” → **75%*** with citations on attendance pages.

---

## Configuration

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Required for chat / API |
| `GROQ_MODEL` | Groq chat model id (must exist on account) |
| `GROQ_MAX_TOKENS` | Max generation tokens |
| `RAG_VERIFY` | `1` (default) enables answer verification |
| `RAG_SIMILARITY_THRESHOLD` | Confidence gate (default `-5.0`) |
| `RAG_CONTEXT_CHARS_PER_CHUNK` | Context trim per chunk |
| `RAG_HISTORY_MAX_MESSAGES` | History window for rewrite |
| `CORS_ORIGINS` | Comma-separated UI origins (defaults to local Vite) |
| `APP_TITLE` / `APP_SUBTITLE` | UI branding via `/api/config` |
| `RAG_PDF_PATH` | PDF path for ingest/embed |

---

## Frontend features

- Multi-chat sidebar (create / rename / delete)
- `localStorage` persistence with `schemaVersion` + sanitization
- Expandable sources; clickable `[Source n]` → highlight source card
- Staged loading UI: Searching → Generating → Verifying (timer-based UX; backend is request/response)

---

## Project layout

| Path | Role |
| --- | --- |
| `ingest.py` | PDF extract + section structure |
| `embed.py` | Chunk packing + embeddings |
| `vector_store.py` | Build FAISS index |
| `bm25.py` | Sparse retrieval |
| `retrieve.py` | Hybrid retrieve + rerank + dedupe |
| `chat.py` | Rewrite, gate, generate, cite |
| `verify.py` | Claim-level PASS / REVISE / FAIL |
| `api_server.py` | FastAPI surface |
| `evaluate_rag.py` | Benchmark harness |
| `diagnose_retrieval.py` | Rank dump for one query |
| `scripts/final_smoke.py` | Product smoke |
| `scripts/hardening_smoke.py` | Validation / error-shape smoke |
| `data/evaluation_set.json` | 19 labeled cases |
| `data/baseline_final.json` | Frozen metrics |
| `frontend/` | React + Vite UI |

---

## Deployment checklist

1. Ship the **frozen** index artifacts (`embeddings.pkl`, `faiss_index.index`) or rebuild them in CI from the same PDFs + `embed.py` / `vector_store.py`.
2. Set `GROQ_API_KEY` and a **verified** `GROQ_MODEL` (hit `/api/health` and one chat before opening to users).
3. Set `CORS_ORIGINS` to the real UI origin(s).
4. Keep `RAG_VERIFY=1` unless you deliberately trade latency for speed.
5. After deploy: `RAG_API_BASE=https://your-api python scripts/final_smoke.py`
6. Confirm UI: citation click, refresh persistence, staged loading.

---

## Troubleshooting

| Problem | What to try |
| --- | --- |
| `Missing GROQ_API_KEY` / 401 | Fix `.env`; restart uvicorn; check `/api/health` → `groq_key_present` |
| Model 404 or empty answers | Change `GROQ_MODEL` to an id listed for your Groq account |
| Stuck on “Verifying…” | Backend still running or model hung; check API logs (UI stages are timers) |
| Index missing | Run `embed.py` then `vector_store.py` |
| UI cannot reach API | uvicorn on **8000**, UI on **5173**; both running |
| Wrong attendance % | Do not “fix” the LLM first—run `diagnose_retrieval.py` (retrieval/chunking issue) |

---

## Licence

Add your organisation’s licence or usage policy if you distribute this repo.
