from sentence_transformers import SentenceTransformer
import pickle
from ingest import extract_pages_from_pdf, clean_text

# Load embedding model  all-MiniLM-L6-v2
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")


def chunk_pages(pages, chunk_size=900, overlap=120):
    """
    pages = [(page_number, page_text), ...]
    """
    chunks = []

    for page_number, page_text in pages:
        page_text = clean_text(page_text)
        start = 0

        while start < len(page_text):
            end = start + chunk_size
            chunk_text = page_text[start:end]

            chunks.append({
                "text": chunk_text,
                "page": page_number
            })

            start = end - overlap

    return chunks


def generate_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


if __name__ == "__main__":
    import os
    import sys

    pdf_path = os.environ.get("RAG_PDF_PATH") or (
        sys.argv[1] if len(sys.argv) > 1 else "data/Academic_Regulations_Hand_Book.pdf"
    )

    # 1. Extract pages
    pages = extract_pages_from_pdf(pdf_path)

    # 2. Chunk pages
    chunks = chunk_pages(pages)

    print(f"Total chunks: {len(chunks)}")

    # 3. Generate embeddings
    embeddings = generate_embeddings(chunks)

    print(f"Embedding shape: {embeddings.shape}")

    # 4. Save chunks + embeddings together
    with open("embeddings.pkl", "wb") as f:
        pickle.dump((chunks, embeddings), f)

    print("✅ Embeddings saved successfully")
