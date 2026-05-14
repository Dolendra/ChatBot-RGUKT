import re
import faiss
import pickle

from sentence_transformers import SentenceTransformer


def _infer_page_from_text(text: str):
    """Best-effort page when legacy chunks are plain strings with `--- Page N ---` markers."""
    if not text:
        return None
    m = re.search(r"---\s*Page\s+(\d+)\s*---", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _normalize_stored_chunk(raw):
    """Support both dict chunks {text, page} and legacy flat strings in embeddings.pkl."""
    if isinstance(raw, dict) and "text" in raw:
        text = str(raw["text"])
        page = raw.get("page")
        if page in (None, "", "—"):
            page = _infer_page_from_text(text) or "—"
        return {"text": text, "page": page}
    if isinstance(raw, str):
        return {"text": raw, "page": _infer_page_from_text(raw) or "—"}
    text = str(raw)
    return {"text": text, "page": _infer_page_from_text(text) or "—"}


def retrieve_chunks(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(_normalize_stored_chunk(chunks[int(idx)]))

    return results



# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load stored chunks and embeddings
with open("embeddings.pkl", "rb") as f:
    chunks, embeddings = pickle.load(f)

# Load FAISS index
index = faiss.read_index("faiss_index.index")


if __name__ == "__main__":
    query = "What are the admission rules for the B.Tech program?"

    results = retrieve_chunks(query)

    for i, chunk in enumerate(results, 1):
        print(f"\n--- Retrieved Chunk {i} (page {chunk['page']}) ---\n")
        print(chunk["text"][:500])

