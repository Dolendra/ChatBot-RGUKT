"""
Diagnose retrieval stages for a query: FAISS, BM25, RRF, CrossEncoder.
Usage: python diagnose_retrieval.py "What is the minimum attendance requirement?"
"""
from __future__ import annotations

import pickle
import sys

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from bm25 import BM25Retriever

QUERY = sys.argv[1] if len(sys.argv) > 1 else (
    "What is the minimum attendance requirement for appearing in examinations?"
)
FOCUS_PAGES = {3, 4, 12, 17}


def main():
    with open("embeddings.pkl", "rb") as f:
        chunks, embeddings = pickle.load(f)
    index = faiss.read_index("faiss_index.index")
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    bm25 = BM25Retriever([c.get("text", "") for c in chunks])
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")
    except Exception as e:
        print("Reranker load failed:", e)
        reranker = None

    print("=" * 70)
    print("QUERY:", QUERY)
    print("=" * 70)

    # FAISS
    qemb = model.encode([QUERY])
    dists, idxs = index.search(qemb, 30)
    faiss_rank = {}
    print("\n--- FAISS top 20 (lower L2 distance = better) ---")
    for rank, (idx, dist) in enumerate(zip(idxs[0], dists[0]), 1):
        idx = int(idx)
        if idx < 0:
            continue
        faiss_rank[idx] = rank
        c = chunks[idx]
        mark = " <<FOCUS" if c.get("page") in FOCUS_PAGES else ""
        print(
            f"  #{rank:02d} idx={idx} page={c.get('page')} dist={float(dist):.4f} "
            f"id={c.get('chunk_id')}{mark}"
        )
        if rank <= 5 or c.get("page") in FOCUS_PAGES:
            print(f"       {c.get('text', '')[:140].replace(chr(10), ' / ')}")

    # BM25
    bm25_hits = bm25.retrieve(QUERY, top_k=30)
    bm25_rank = {}
    print("\n--- BM25 top 20 (higher score = better) ---")
    for rank, (idx, score) in enumerate(bm25_hits[:20], 1):
        idx = int(idx)
        bm25_rank[idx] = rank
        c = chunks[idx]
        mark = " <<FOCUS" if c.get("page") in FOCUS_PAGES else ""
        print(
            f"  #{rank:02d} idx={idx} page={c.get('page')} score={float(score):.4f} "
            f"id={c.get('chunk_id')}{mark}"
        )
        if rank <= 5 or c.get("page") in FOCUS_PAGES:
            print(f"       {c.get('text', '')[:140].replace(chr(10), ' / ')}")

    # RRF
    rrf = {}
    for rank, idx in enumerate(idxs[0][:20], 1):
        idx = int(idx)
        if idx < 0:
            continue
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
    for rank, (idx, _score) in enumerate(bm25_hits[:20], 1):
        idx = int(idx)
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
    sorted_rrf = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\n--- RRF top 20 ---")
    for rank, (idx, score) in enumerate(sorted_rrf, 1):
        c = chunks[idx]
        mark = " <<FOCUS" if c.get("page") in FOCUS_PAGES else ""
        print(
            f"  #{rank:02d} idx={idx} page={c.get('page')} rrf={score:.6f} "
            f"faiss_rank={faiss_rank.get(idx, '-')} bm25_rank={bm25_rank.get(idx, '-')} "
            f"id={c.get('chunk_id')}{mark}"
        )

    # Focus page summary: any chunk on those pages in pools?
    print("\n--- FOCUS PAGE PRESENCE ---")
    for page in sorted(FOCUS_PAGES):
        page_idxs = [i for i, c in enumerate(chunks) if c.get("page") == page]
        in_faiss = [i for i in page_idxs if i in faiss_rank]
        in_bm25 = [i for i in page_idxs if i in bm25_rank]
        in_rrf = [i for i in page_idxs if i in rrf]
        print(
            f"  page {page}: chunks={len(page_idxs)} "
            f"in_faiss30={len(in_faiss)} in_bm2530={len(in_bm25)} in_rrf={len(in_rrf)}"
        )
        # Show best attendance-related chunk on page
        for i in page_idxs:
            text = chunks[i].get("text", "")
            if "75%" in text or "40%" in text or "attendance" in text.lower():
                print(
                    f"    idx={i} faiss={faiss_rank.get(i,'-')} bm25={bm25_rank.get(i,'-')} "
                    f"rrf_rank={[r for r,(ix,_) in enumerate(sorted_rrf,1) if ix==i] or '-'} "
                    f"| {text[:160].replace(chr(10),' / ')}"
                )

    # Rerank RRF candidates
    if reranker and sorted_rrf:
        cand_idxs = [idx for idx, _ in sorted_rrf]
        pairs = [[QUERY, chunks[i].get("text", "")] for i in cand_idxs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(cand_idxs, scores), key=lambda x: x[1], reverse=True)
        print("\n--- CrossEncoder rerank of RRF-20 ---")
        for rank, (idx, score) in enumerate(ranked, 1):
            c = chunks[idx]
            mark = " <<FOCUS" if c.get("page") in FOCUS_PAGES else ""
            print(
                f"  #{rank:02d} idx={idx} page={c.get('page')} ce={float(score):.4f} "
                f"id={c.get('chunk_id')}{mark}"
            )
            if c.get("page") in FOCUS_PAGES or rank <= 8:
                print(f"       {c.get('text', '')[:140].replace(chr(10), ' / ')}")

        # Specifically score page-3 75% chunk if not in pool
        print("\n--- Direct CE score for page-3 75% chunks (even if outside pool) ---")
        for i, c in enumerate(chunks):
            if c.get("page") == 3 and "75%" in c.get("text", ""):
                ce = float(reranker.predict([[QUERY, c["text"]]])[0])
                print(f"  idx={i} id={c['chunk_id']} ce={ce:.4f}")
                print(f"  {c['text'][:220].replace(chr(10), ' / ')}")


if __name__ == "__main__":
    main()
