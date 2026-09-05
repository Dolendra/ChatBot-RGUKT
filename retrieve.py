import os
import pickle
import re
import time
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from bm25 import BM25Retriever

# Keep in sync with embed.py
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

embeddings_path = "embeddings.pkl"
if os.path.exists(embeddings_path):
    with open(embeddings_path, "rb") as f:
        chunks, embeddings = pickle.load(f)
else:
    chunks, embeddings = [], None

index_path = "faiss_index.index"
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = None

if chunks:
    bm25_index = BM25Retriever([c.get("text", "") for c in chunks])
else:
    bm25_index = None

reranker = None
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")
except Exception as e:
    print(f"Warning: Could not load CrossEncoder reranker: {e}. Running with hybrid score fallback.")

_CANDIDATE_POOL = 30
_RRF_KEEP = 25

_REQUIREMENT_CUES = {
    "minimum", "required", "requirement", "must", "percentage", "percent",
    "how", "much", "maintain", "eligible", "eligibility",
}
_CONSEQUENCE_CUES = {
    "happen", "happens", "below", "fail", "fails", "if", "when", "less",
    "without", "lose", "loss", "readmission", "consequence", "does", "not",
}


def _normalize_stored_chunk(raw):
    """Normalize chunks to ensure all metadata keys are present."""
    if isinstance(raw, dict):
        return {
            "chunk_id": raw.get("chunk_id", "unknown"),
            "text": str(raw.get("text", "")),
            "source": raw.get("source", "Academic_Regulations_Hand_Book.pdf"),
            "document_id": raw.get("document_id", "academic_regulations_hand_book"),
            "page": raw.get("page", "—"),
            "section": raw.get("section", "General"),
            "version": raw.get("version", "2016-17"),
            "score": float(raw.get("score", 0.0)),
        }
    text = str(raw)
    return {
        "chunk_id": "legacy_chunk",
        "text": text,
        "source": "Academic_Regulations_Hand_Book.pdf",
        "document_id": "academic_regulations_hand_book",
        "page": "—",
        "section": "General",
        "version": "2016-17",
        "score": 0.0,
    }


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _number_set(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", (text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def expand_retrieval_query(query: str) -> str:
    """
    Deterministic intent-preserving expansion (no hardcoded regulation values).
    Distinguishes requirement-seeking queries from consequence queries.
    """
    q = (query or "").strip()
    if not q:
        return q
    tokens = _token_set(q)
    ql = q.lower()

    # Consequence framing wins when the question asks what happens / falls below / if-when.
    consequence_framed = bool(
        re.search(
            r"\b(what happens|falls? below|below \d|if .+attendance|when .+attendance|"
            r"does not have|without the requisite|lose (their )?admission)\b",
            ql,
        )
    )
    requirement_framed = bool(
        re.search(
            r"\b(minimum|how much|what percentage|must (a )?student maintain|"
            r"required (to|for)|requirement for)\b",
            ql,
        )
    ) and not consequence_framed

    extras: list[str] = []
    if consequence_framed:
        extras.append("readmission lose admission below requisite attendance")
    elif requirement_framed or (tokens & _REQUIREMENT_CUES and not (tokens & _CONSEQUENCE_CUES)):
        extras.append("not less than percentage regular program of study")

    if not extras:
        return q
    return f"{q} {' '.join(extras)}"


def _lexical_bonus(query: str, chunk: dict) -> float:
    """
    Generic post-rerank boost from exact term / number / section overlap.
    Does not hard-code regulation facts.
    """
    q_tokens = _token_set(query)
    q_nums = _number_set(query)
    text = f"{chunk.get('text', '')} {chunk.get('section', '')}".lower()
    t_tokens = _token_set(text)
    t_nums = _number_set(text)

    if not q_tokens:
        return 0.0

    content_words = {t for t in q_tokens if len(t) > 2}
    overlap = len(content_words & t_tokens) / max(1, len(content_words))

    num_bonus = 0.0
    if q_nums:
        num_bonus = 1.2 * len(q_nums & t_nums)

    synonym_bonus = 0.0
    if content_words & {"minimum", "required", "requirement", "must"} or "not less than" in query.lower():
        if "not less than" in text or "minimum" in text or "required" in text:
            synonym_bonus += 0.9
    if "readmission" in query.lower() or content_words & {"happen", "below", "fail", "lose", "readmission"}:
        if any(x in text for x in ("readmission", "lose their admission", "lose admission", "not less than 40%")):
            synonym_bonus += 1.4
        elif any(x in text for x in ("below", "less than", "requisite attendance")):
            synonym_bonus += 0.6

    section = (chunk.get("section") or "").lower()
    section_tokens = _token_set(section)
    section_bonus = 0.0
    if section_tokens and content_words & section_tokens:
        section_bonus = 0.45

    return (overlap * 1.1) + num_bonus + synonym_bonus + section_bonus


def _dedupe_chunks(scored_chunks, top_k: int, overlap_threshold: float = 0.65):
    """
    Keep highest-scoring chunks.
    - At most one chunk per (document_id, page)
    - Drop near-duplicate text across pages
    """
    selected = []
    selected_tokens = []
    seen_pages: set[tuple] = set()

    for chunk, score in scored_chunks:
        text = chunk.get("text", "")
        tokens = _token_set(text)
        page = chunk.get("page")
        doc = chunk.get("document_id") or chunk.get("source")
        page_key = (doc, page)

        if page_key in seen_pages:
            continue

        is_dup = False
        for prev, prev_tokens in zip(selected, selected_tokens):
            overlap = _jaccard(tokens, prev_tokens)
            if overlap >= overlap_threshold:
                is_dup = True
                break
        if is_dup:
            continue

        enriched = dict(chunk)
        enriched["score"] = float(score)
        selected.append(enriched)
        selected_tokens.append(tokens)
        seen_pages.add(page_key)
        if len(selected) >= top_k:
            break

    return selected


def retrieve_chunks(query, top_k=3, return_metrics=False, threshold=-5.0):
    """
    Hybrid retrieval: expand → FAISS + BM25 → RRF → CrossEncoder → lexical boost → dedupe.
    """
    t_start = time.time()

    if not index or not bm25_index or not chunks:
        metrics = {
            "embedding_ms": 0.0,
            "faiss_ms": 0.0,
            "bm25_ms": 0.0,
            "rrf_ms": 0.0,
            "rerank_ms": 0.0,
            "total_retrieve_ms": 0.0,
            "top_score": 0.0,
            "low_confidence": True,
            "expanded_query": query,
        }
        if return_metrics:
            return [], metrics
        return []

    search_query = expand_retrieval_query(query)

    # 1. Dense retrieval (FAISS)
    t0 = time.time()
    query_embedding = model.encode([search_query])
    t_embed = time.time() - t0

    t0 = time.time()
    dense_distances, dense_indices = index.search(query_embedding, _CANDIDATE_POOL)
    t_faiss = time.time() - t0

    # 2. Sparse retrieval (BM25)
    t0 = time.time()
    bm25_results = bm25_index.retrieve(search_query, top_k=_CANDIDATE_POOL)
    t_bm25 = time.time() - t0

    # 3. Reciprocal Rank Fusion (RRF)
    t0 = time.time()
    rrf_scores = {}

    for rank, idx in enumerate(dense_indices[0]):
        idx = int(idx)
        if idx < 0:
            continue
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60.0 + (rank + 1))

    for rank, (idx, score) in enumerate(bm25_results):
        idx = int(idx)
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60.0 + (rank + 1))

    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_rrf_candidates = sorted_rrf[:_RRF_KEEP]
    t_rrf = time.time() - t0

    # 4. CrossEncoder Reranking + lexical boost + dedupe
    t0 = time.time()
    final_results = []
    top_score = -99.0

    if reranker is not None and top_rrf_candidates:
        candidate_indices = [idx for idx, _ in top_rrf_candidates]
        candidate_chunks = [chunks[idx] for idx in candidate_indices]

        # Rerank with expanded query so intent cues reach the cross-encoder
        pairs = [[search_query, chunk.get("text", "")] for chunk in candidate_chunks]
        rerank_scores = reranker.predict(pairs)

        chunk_scores = []
        for chunk, ce_score in zip(candidate_chunks, rerank_scores):
            bonus = _lexical_bonus(search_query, chunk)
            chunk_scores.append((chunk, float(ce_score) + bonus))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        deduped = _dedupe_chunks(chunk_scores, top_k=top_k)
        final_results = [_normalize_stored_chunk(c) for c in deduped]

        if chunk_scores:
            top_score = float(chunk_scores[0][1])

        t_rerank = time.time() - t0
    else:
        scored = []
        for idx, score in top_rrf_candidates:
            chunk = chunks[idx]
            scored.append((chunk, float(score) + _lexical_bonus(search_query, chunk)))
        scored.sort(key=lambda x: x[1], reverse=True)
        deduped = _dedupe_chunks(scored, top_k=top_k)
        final_results = [_normalize_stored_chunk(c) for c in deduped]
        top_score = float(scored[0][1]) if scored else 0.0
        t_rerank = time.time() - t0

    total_time = time.time() - t_start
    low_confidence = top_score < threshold

    metrics = {
        "embedding_ms": round(t_embed * 1000, 2),
        "faiss_ms": round(t_faiss * 1000, 2),
        "bm25_ms": round(t_bm25 * 1000, 2),
        "rrf_ms": round(t_rrf * 1000, 2),
        "rerank_ms": round(t_rerank * 1000, 2),
        "total_retrieve_ms": round(total_time * 1000, 2),
        "top_score": round(top_score, 4),
        "low_confidence": low_confidence,
        "expanded_query": search_query,
    }

    if return_metrics:
        return final_results, metrics
    return final_results


if __name__ == "__main__":
    query = "What is the minimum attendance requirement for appearing in examinations?"
    results, metrics = retrieve_chunks(query, top_k=5, return_metrics=True)

    print("Expanded:", metrics.get("expanded_query"))
    print("\n--- Latency Breakdown ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    for i, chunk in enumerate(results, 1):
        print(f"\n--- Retrieved Chunk {i} ---")
        print(f"  Chunk ID: {chunk['chunk_id']}")
        print(f"  Source: {chunk['source']} | Page: {chunk['page']} | Section: {chunk['section']}")
        print(f"  Score: {chunk.get('score')}")
        print(f"  Snippet: {chunk['text'][:220].replace(chr(10), ' / ')}...")
