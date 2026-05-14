import pickle

import faiss


def build_index(embeddings_path="embeddings.pkl", index_path="faiss_index.index"):
    with open(embeddings_path, "rb") as f:
        chunks, embeddings = pickle.load(f)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return len(chunks), index.ntotal


if __name__ == "__main__":
    n_chunks, n_vectors = build_index()
    print("FAISS index built")
    print("Chunks:", n_chunks, "| Vectors in index:", n_vectors)
