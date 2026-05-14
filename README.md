# Organisation RAG Chatbot

A small **Retrieval-Augmented Generation (RAG)** assistant that answers questions **only from your own PDF documents**, with **page-level citations**. It runs on your machine for indexing; answers use the **Groq** cloud API for the language model.

If you have never used Python or APIs before, follow the steps in order under [Quick start](#quick-start).

---

## What this is (in plain language)

1. **You** put internal PDFs (handbooks, policies, FAQs) in the `data/` folder.
2. **The program** reads the PDFs, splits them into overlapping text chunks, and turns each chunk into a **vector** (a list of numbers that captures meaning).
3. **When someone asks a question**, the program finds the chunks whose vectors are closest to the question’s vector (**semantic search**).
4. **Those chunks** are sent to a **large language model (LLM)** with strict instructions: answer **only** from that text, cite sources, and say **“I don’t know”** if the text does not contain the answer.

So the chatbot is **grounded in your documents**, not in general internet knowledge.

---

## High-level architecture (for your organisation)

| Stage | What happens | Main file |
|--------|----------------|-----------|
| **Ingest** | PDF → text per page, light cleaning | `ingest.py` |
| **Embed** | Pages → chunks → embeddings → `embeddings.pkl` | `embed.py` |
| **Index** | Embeddings → FAISS vector index on disk | `vector_store.py` |
| **Retrieve** | Question → embedding → top similar chunks | `retrieve.py` |
| **Answer** | Chunks + question → Groq LLM → cited answer | `chat.py` |
| **Web API + UI** | Browser → FastAPI → same RAG flow → structured JSON | `api_server.py`, `frontend/` |

**Data flow (one line):**  
`PDF` → `chunks + vectors` → `FAISS index` → `retrieve top-k` → `LLM with context` → `answer`.

**Typical roles**

- **Content owner:** Places or updates PDFs in `data/`, then re-runs embed + index when documents change.
- **IT / security:** Manages API keys, who can run the chatbot, and where PDFs may live (this repo does not include auth or multi-user hosting; those would be added for a production internal deployment).

**What this project does *not* include (yet)**

- No Teams/Slack integration (browser UI + local API only).
- No automatic “live” sync when a PDF changes (re-run `embed.py` and `vector_store.py` after updates).
- No user login or audit log (add those if you expose it beyond a trusted machine).

---

## Quick start

### 1. Prerequisites

- **Python 3.10+** installed.
- A **Groq** account and API key: [Groq Console – API keys](https://console.groq.com/keys).

### 2. Clone or copy this project

Open a terminal in the project folder (the same folder as this `README.md`).

### 3. Create a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure the API key

Copy the example env file and edit it:

```powershell
copy .env.example .env
```

Open `.env` in an editor and set:

```env
GROQ_API_KEY=your_key_here
```

The chat script loads `.env` automatically (`python-dotenv`).

### 5. Add your PDF

Put at least one PDF under `data/`. By default the scripts expect:

`data/Academic_Regulations_Hand_Book.pdf`

To use another file:

- **Embed step:** pass the path as the first argument, or set `RAG_PDF_PATH` in the environment.

Examples:

```powershell
$env:RAG_PDF_PATH = "data/Your_Handbook.pdf"
python embed.py
```

```powershell
python embed.py "data/Your_Handbook.pdf"
```

### 6. Build embeddings and the search index

Run **in order**:

```powershell
python embed.py
python vector_store.py
```

This creates:

- `embeddings.pkl` — chunks + their vectors  
- `faiss_index.index` — fast similarity search structure  

Both files are listed in `.gitignore` so they are not committed by mistake (they can be large and may contain sensitive text).

### 7. Chat

```powershell
python chat.py
```

Type questions; type `exit` to quit.

### 8. Web chat (Vite + React)

Use this for a **ChatGPT-style** interface: markdown answers, typing indicator, and expandable “retrieved context” per reply.

**Terminal A — API (from project root, venv active):**

```powershell
pip install -r requirements.txt
uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

**Terminal B — frontend:**

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The dev server proxies `/api` to the backend on port **8000**.

If the page cannot reach the API, confirm `uvicorn` is running and that `embeddings.pkl` and `faiss_index.index` exist (same as CLI chat).

Optional: set `CORS_ORIGINS` in `.env` (comma-separated URLs) if you host the UI on another origin.

---

## Optional checks

**Test PDF extraction only**

```powershell
python ingest.py
```

Or with a specific PDF:

```powershell
python ingest.py "data/Your_Handbook.pdf"
```

**Test retrieval only (no LLM)**

```powershell
python retrieve.py
```

---

## Configuration summary

| Variable / argument | Purpose |
|---------------------|---------|
| `GROQ_API_KEY` in `.env` | Required for `chat.py` and `api_server.py` |
| `GROQ_MODEL`, `GROQ_MAX_TOKENS` in `.env` | Optional overrides for the Groq model and max reply length |
| `RAG_PDF_PATH` or `python embed.py <path>` | Which PDF to index |
| `retrieve_chunks` `top_k` (CLI) / request body (web) | How many chunks to send to the model |
| Groq model | Defaults in `chat.py`; override with `GROQ_MODEL` — see [Groq models](https://console.groq.com/docs/models) |

---

## Updating documents after policies change

1. Replace or add PDFs under `data/`.
2. Run `python embed.py` (and point it at the right PDF if needed).
3. Run `python vector_store.py` again to rebuild `faiss_index.index`.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `Missing GROQ_API_KEY` | Create `.env` in the project root (same folder as `api_server.py`) or export `GROQ_API_KEY` in the terminal before `uvicorn`. |
| Groq `401 invalid_api_key` | Key must be valid on [Groq Console](https://console.groq.com/keys). In `.env` use `GROQ_API_KEY=gsk_...` on one line (no spaces around `=`). Save as UTF-8, restart `uvicorn`. Open `http://127.0.0.1:8000/api/health` — `groq_key_present` should be `true`. |
| File not found for PDF | Check the path under `data/` and `RAG_PDF_PATH` / CLI argument. |
| Error loading index | Run `embed.py` then `vector_store.py` so both `embeddings.pkl` and `faiss_index.index` exist. |
| Web UI: network / failed fetch | Start `uvicorn` on port 8000 before `npm run dev`, and keep both terminals open. |
| First run is slow | The embedding model downloads once; later runs reuse the cache. |

---

## Licence and usage

Add your organisation’s licence or usage policy here if you distribute this repo.

---

## Summary

This repository is a **minimal internal RAG pipeline**: PDFs → embeddings → FAISS → Groq LLM, with **source citations** and **refusal** when the context does not support an answer. A **React + Vite** UI and **FastAPI** server are included for local demos; add auth and hosting when you move from a prototype to a shared service.
