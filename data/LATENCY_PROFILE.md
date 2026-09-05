# Final latency profile (from frozen baseline_final.json)

Source: `data/baseline_final.json` (19-question suite, post-verification).

## Average component latency

| Stage | Avg ms | Share of total |
| --- | ---: | ---: |
| Embedding | 25.0 | 0.1% |
| FAISS dense | 1.4 | <0.1% |
| BM25 | 16.3 | 0.1% |
| RRF | 0.1 | <0.1% |
| CrossEncoder rerank | 550.9 | 2.7% |
| LLM generation (draft) | 7556.0 | 36.8% |
| Answer verification | 12405.0 | 60.4% |
| **Total** | **20554.8** | **100%** |

Intent expansion is local/deterministic. Conversational LLM rewrite (follow-ups only) is included in generation when used.

## Interpretation

1. **Retrieval is cheap** (~0.6s). Ranking quality is already strong; do not chase FAISS/BM25 micro-optimizations first.
2. **Verification dominates (~12.4s / 60%)**, then **draft generation (~7.6s / 37%)**.
3. Both heavy stages are primarily **Groq API wait / token generation**, not local CPU post-processing.
4. Product: keep staged loading UX. Optional later: selective verification for high-risk numeric/regulation queries only (not in this freeze).

## Refresh

```bash
python evaluate_rag.py
# copy data/evaluation_results.json → data/baseline_final.json when complete
```
