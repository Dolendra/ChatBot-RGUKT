# RAG Upgrade Plan: RRF Hybrid Search, Reranking, Calibrated Thresholds, and Dual-Stage Evaluation

This plan outlines the enhancements to transform the current RAG prototype into a robust, evaluated, and production-like system ready for placement demonstrations.

## User Review Required

> [!IMPORTANT]
> - **Reranker Model Download**: The CrossEncoder reranker uses `cross-encoder/ms-marco-MiniLM-L-2-v2` (~80MB). On the first run, the system will automatically download it. A fallback to normalized dense + BM25 scores is implemented in case the Hugging Face Hub is unreachable.
> - **RRF Constant ($k=60$)**: We will use the standard Reciprocal Rank Fusion constant $k=60$ to combine FAISS dense and BM25 sparse search rankings.
> - **Threshold Calibration**: Rather than hardcoding a similarity threshold, we will write a script to evaluate different threshold values on a calibration set, checking for False Accepts, False Rejects, and Abstention Accuracy.
> - **LLM-as-a-Judge**: We will implement a lightweight LLM prompt to Groq to grade generation faithfulness and correctness.

## Proposed Changes

We will restructure the pipeline to support multiple files, extract metadata, combine dense and keyword search, rerank candidate documents, evaluate retrieval accuracy, and present the results in a premium developer dashboard.

---

### Ingestion & Chunking Upgrade

We will update the ingestion and embedding pipeline to scan all PDFs in `data/`, extract their titles, versions, and propagate section titles page-by-page.

#### [MODIFY] [ingest.py](file:///d:/chatbot-RAG/ingest.py)
- Update text extraction to record PDF names and detect structural headings (e.g. `X. RULES OF PROMOTION`, `VII. GRADING SYSTEM`) to assign section names to pages.

#### [MODIFY] [embed.py](file:///d:/chatbot-RAG/embed.py)
- Update `chunk_pages` to output a list of enriched chunks containing:
  ```json
  {
    "chunk_id": "academic_regulations_hand_book_p12_c03",
    "text": "...",
    "source": "Academic_Regulations_Hand_Book.pdf",
    "document_id": "academic_regulations_hand_book",
    "page": 12,
    "section": "RULES OF PROMOTION",
    "version": "2016-17"
  }
  ```
- Scan all `.pdf` files inside `data/` instead of just a single hardcoded path, allowing multi-document search.

---

### Retrieval Engine Enhancements

We will add BM25 keyword search, merge results with FAISS dense retrieval, apply CrossEncoder reranking, and implement a confidence threshold to prevent hallucinations.

#### [NEW] [bm25.py](file:///d:/chatbot-RAG/bm25.py)
- A self-contained, pure-Python BM25 retriever class `BM25Retriever`. This ensures no external dependencies and allows instant keyword matching.

#### [MODIFY] [retrieve.py](file:///d:/chatbot-RAG/retrieve.py)
- Initialize both the FAISS index and the `BM25Retriever` on startup.
- Implement Reciprocal Rank Fusion (RRF) to merge Dense and BM25 results.
  1. Retrieve top 20 candidate chunks with FAISS.
  2. Retrieve top 20 candidate chunks with BM25.
  3. Merge and compute $RRF(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$.
  4. Take the top 20 candidates.
- Load `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")` to rerank the top 20 merged candidates.
- Track latency per step: embedding, FAISS search, BM25 search, RRF fusion, reranking.
- Implement an empirically calibrated threshold on reranking score. If the best score is below the threshold, classify it as low confidence.

---

### Generation & Conversational Memory

We will update the generation prompt to incorporate enriched citations (document name, section, page) and handle low-confidence retrieval by abstaining.

#### [MODIFY] [chat.py](file:///d:/chatbot-RAG/chat.py)
- Pass metadata down to the LLM prompt. Instruct the model to cite specific pages and document sections.
- If the best retrieved chunk fails the similarity threshold, bypass the LLM and return: *"I couldn't find any relevant information in the RGUKT documents."*
- Measure and return generation latency.

---

### Evaluation System

We will build an automated evaluation dataset and script to measure Recall@K, generation metrics (faithfulness, correctness), and component-level latency.

#### [NEW] [evaluation_set.json](file:///d:/chatbot-RAG/data/evaluation_set.json)
- Store a dataset of 30+ questions about RGUKT regulations, each specifying expected keywords, expected page numbers, expected chunk IDs, and reference answers.
- Categories: Simple factual, Regulations, Exact identifiers, Numerical, Multi-section, Out-of-domain.

#### [NEW] [evaluate_rag.py](file:///d:/chatbot-RAG/evaluate_rag.py)
- Create a CLI script to run evaluation:
  - Query the retrieval engine with the evaluation questions.
  - Calculate **Recall@1**, **Recall@3**, and **Recall@5** (by checking if the ground-truth page/chunk ID matches the retrieved contexts).
  - Grade **Faithfulness** and **Answer Correctness** using Groq LLM-as-a-judge.
  - Track component-level latency (Embedding, FAISS, BM25, RRF, Reranking, Generation).
  - Save output to `data/evaluation_results.json`.

---

### API & Developer Dashboard UI

We will expose the evaluation metrics via FastAPI and add a premium developer dashboard tab in the React frontend.

#### [MODIFY] [api_server.py](file:///d:/chatbot-RAG/api_server.py)
- Add endpoints:
  - `GET /api/evaluate`: Returns the latest evaluation run results.
  - `POST /api/evaluate/run`: Triggers a live evaluation run and saves the report.
- Include section, document titles, versions, chunk IDs, and latency details in the response returned by `POST /api/chat`.

#### [MODIFY] [App.jsx](file:///d:/chatbot-RAG/frontend/src/App.jsx)
- Redesign the chat window header to include a view toggle between "Chat" and "Evaluation Dashboard".
- Build an elegant Dashboard UI containing:
  - Metrics Cards: Recall@1, Recall@3, Recall@5, Faithfulness, Correctness, and component-level latency breakdown.
  - Test Cases Table: Displays each test question, category, status, latency breakdown, and matching pages.
- Redesign the citation details to display clean icons, document source name, section headings, page numbers, and document versions.

---

## Verification Plan

### Automated Tests
- Run `evaluate_rag.py` to verify the retrieval metrics pipeline works and records Recall@K.
- Assert that query latency remains within acceptable limits (< 2s on average for retrieval + reranking).

### Manual Verification
- Re-index documents with `embed.py` and `vector_store.py`.
- Query out-of-domain questions (e.g. "Who won the 2026 World Cup?") and verify that the system correctly triggers the similarity threshold fallback.
- Test normal queries in the UI and verify that citation badges (source name, section, page, version) render correctly.
- Toggle to the developer dashboard and run a live evaluation. Verify metrics update dynamically.
