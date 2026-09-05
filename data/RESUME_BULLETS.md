# Resume bullets (RGUKT Academic RAG)

Use 2–3 of these; keep numbers exact to `data/baseline_final.json`.

---

**Option A — full stack (recommended)**

- Built a grounded academic RAG chatbot (PDF → hybrid FAISS+BM25 → RRF → CrossEncoder → Groq LLM) with claim-level answer verification, page citations, and a React/FastAPI UI; froze a 19-question eval at **R@1 94.7%**, **R@3/5 100%**, **MRR 96.5%**, faithfulness **82.1%**, OOD refusal **100%**.

**Option B — retrieval focus**

- Designed structure-aware chunking and hybrid retrieval (dense + BM25 + RRF + cross-encoder + page dedupe) that fixed attendance ranking failures; confidence gate abstains on low evidence instead of hallucinating.

**Option C — evaluation / product hardening**

- Instrumented end-to-end metrics (recall, faithfulness, citation grounding, latency ~20.6s with verify ~60%) and shipped API hardening + smoke tests so model/config failures (e.g. unavailable Groq ids) are separated from RAG regressions.

---

### One-liner for “tell me about a project”

> I built an academic regulations Q&A system that only answers from PDFs. The hard part wasn’t the chat UI—it was getting the right page retrieved for numeric rules like 75% attendance, then verifying the LLM didn’t invent numbers. I measure that with a frozen 19-question suite rather than vibes.
