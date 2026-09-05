# Interview walkthrough — RGUKT Academic RAG

Use this as a spoken outline. Numbers from `data/baseline_final.json`.  
**Clarify early:** Groq is the **chat LLM**, not the embedding model. Embeddings = `paraphrase-MiniLM-L3-v2`.

---

## 60-second pitch

I built a RAG assistant over RGUKT academic regulations. PDFs are chunked with section awareness, embedded locally, and searched with hybrid retrieval (FAISS + BM25 → RRF → CrossEncoder). Groq generates the answer; a verifier checks claims against evidence. The UI is React + FastAPI with citations and chat history. Quality is frozen on a 19-question benchmark: R@1 ~95%, R@3 100%, OOD refusal 100%.

---

## Component “why / alternatives / failure / measure / scale”

### PDF → ingest
- **Why:** Need clean page text + section headings for citations and ranking boosts.
- **Alternatives:** OCR-heavy pipelines, unstructured.io, LlamaParse.
- **Failed:** Messy page joins glued **75% attendance** to unrelated seat-allocation text → wrong answers despite “good” embeddings.
- **Measure:** Attendance regression + retrieval diagnostics.
- **Scale:** Parallel page extract; document registry with `document_id` / version.

### Chunking
- **Why:** Structure-aware blocks beat naive fixed windows for regulations.
- **Alternatives:** Fixed 512/overlap, semantic splitters, parent-document retrieval.
- **Failed:** Over-large chunks mixed rules; under-chunking lost numbers.
- **Measure:** R@1 on numeric / section-labeled eval cases.
- **Scale:** Versioned chunk manifests; rebuild index on schema change only.

### Embedding (`paraphrase-MiniLM-L3-v2`)
- **Why:** Fast local CPU model; good enough for handbook-scale corpus.
- **Alternatives:** larger MiniLM/MPNet, OpenAI/Voyage embeddings, late interaction (ColBERT).
- **Tradeoff:** Tiny model → cheap; may miss paraphrase nuance (compensated by BM25 + CE).
- **Scale:** Batch embed; swap model → full re-index + re-eval.

### FAISS
- **Why:** Exact L2 search fine at current size (`IndexFlatL2`).
- **Alternatives:** HNSW/IVF, Qdrant/Pinecone/pgvector.
- **Scale:** Move to ANN + metadata filters when docs ≫ tens of thousands of chunks.

### BM25
- **Why:** Regulations are **number- and phrase-heavy** (“75%”, “40%”); dense alone missed lexical hits.
- **Alternatives:** Elasticsearch, Lucene, SPLADE.
- **Measure:** Side-by-side FAISS vs BM25 vs hybrid via `diagnose_retrieval.py`.

### RRF
- **Why:** Robust fusion without score calibration between FAISS distances and BM25.
- **Alternatives:** weighted sum, learned fusion.
- **Scale:** Keep RRF until you have click/eval labels for learning to rank.

### CrossEncoder
- **Why:** Rerank top fused candidates with query–doc pairwise scores.
- **Alternatives:** larger CE, LLM-as-reranker (costly).
- **Latency:** ~0.55s avg in baseline—acceptable vs multi-second LLM.

### Query rewriting
- **Local expand:** Intent cues (requirement vs consequence)—deterministic, free.
- **LLM rewrite:** Only short/follow-up turns with history → standalone search query.
- **Why split:** Don’t pay LLM latency on every query; follow-ups like “below that?” need context.

### Confidence gate
- **Why:** Low top score → abstain (`couldn't find…`) without calling the LLM.
- **Measure:** OOD 100%, false refusal 0% on suite.

### Groq LLM
- **Why:** Fast/cheap chat API for draft (+ verify).
- **Not used for:** embeddings.
- **Failed in dev:** Unavailable / empty model ids (`llama` 404, some `gpt-oss` empty)—**config**, not retrieval.
- **Deploy lesson:** Validate `GROQ_MODEL` against the account before users hit it.

### Verification
- **Why:** Faithfulness on numeric rules; PASS / REVISE / FAIL.
- **Cost:** ~60% of ~20.6s latency.
- **Future:** Selective verify on numeric/regulation queries only.

### Citations + UI
- **Why:** Trust—`[Source n]` maps to page cards; click-to-highlight.
- **API:** User-facing errors only; no chat body in logs; CORS locked down.

---

## Metrics story (say this)

> I don’t argue quality from demos alone. Nineteen labeled questions use **page/section/keyword** evidence. After the attendance ranking fix + verifier: R@1 94.7%, R@3/5 100%, MRR 96.5%, faithfulness 82%, citation IDs 100% valid, OOD refusal 100%. Latency is dominated by verification and generation, not retrieval—so I wouldn’t micro-optimize FAISS next.

---

## Hardest bug (tell as a story)

Attendance answers were wrong because retrieval returned the wrong neighborhood and/or chunks mixed 75% with unrelated text. Fix path: structure-aware chunking → hybrid + boosts → page dedupe → verify numbers. Regression: “minimum attendance?” → **75%**.

---

## What you’d do next (scaling / product)

1. Deploy frozen build; smoke with `final_smoke.py` against prod API.
2. Selective verification to cut latency.
3. Auth + audit log if multi-user.
4. Multi-doc versioning and “re-index when PDF changes” job.
5. Only then consider larger embedding / managed vector DB.

---

## Questions to expect

| Question | Short answer |
| --- | --- |
| Why hybrid? | Dense for paraphrase; BM25 for exact % and titles. |
| Why not only LLM? | Hallucinated regulations; need evidence + cite. |
| Why verify if prompt says “only use context”? | Prompts aren’t enough for numbers; measure faithfulness. |
| Why is P@5 low? | Intentional—care about top ranks / small `top_k`, not flooding context. |
| Embeddings on Groq? | **No.** Local SentenceTransformer; Groq = generation/verify. |
| How test? | `evaluate_rag.py`, `final_smoke.py`, manual UI citation/persistence. |
