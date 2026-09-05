import pickle
import re
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from ingest import extract_pages_from_pdf, clean_text, split_into_blocks

# Keep in sync with retrieve.py
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"
model = SentenceTransformer(EMBEDDING_MODEL)


def extract_version_from_filename(filename):
    m = re.search(r"\b(20\d{2}[\-_]\d{2,4}|20\d{2})\b", filename)
    if m:
        return m.group(1).replace("_", "-")
    return "2016-17"


def _hard_split_block(block: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(block):
        end = min(start + max_chars, len(block))
        if end < len(block):
            window = block[start:end]
            br = window.rfind("\n")
            if br < max_chars // 3:
                br = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
                if br != -1:
                    br += 1
            if br < max_chars // 3:
                br = window.rfind(" ")
            if br > max_chars // 3:
                end = start + br
        piece = block[start:end].strip()
        if piece:
            pieces.append(piece)
        next_start = end - 120 if end < len(block) else end
        if next_start <= start:
            next_start = end
        start = next_start
    return pieces


def _pack_block_dicts(blocks: list[dict], max_chars=900, min_chars=80) -> list[dict]:
    """
    Pack structural blocks into chunks.
    Never merge across major section boundaries (is_section_start=True).
    """
    if not blocks:
        return []

    packed: list[dict] = []
    i = 0
    while i < len(blocks):
        parts: list[str] = []
        size = 0
        section = blocks[i].get("section", "General")

        # Major section heading starts a fresh pack group
        if blocks[i].get("is_section_start") and parts:
            pass  # handled by loop break below

        while i < len(blocks):
            b = blocks[i]
            text = b.get("text", "")
            # Hard break before a new major section once we already have content
            if parts and b.get("is_section_start"):
                break

            if not parts and len(text) > max_chars:
                for piece in _hard_split_block(text, max_chars):
                    packed.append({"text": piece, "section": b.get("section", section)})
                i += 1
                break

            extra = len(text) + (1 if parts else 0)
            if parts and size + extra > max_chars:
                break

            if not parts:
                section = b.get("section", section)
            parts.append(text)
            size += extra
            i += 1

        if parts:
            combined = "\n".join(parts).strip()
            if (
                packed
                and len(combined) < min_chars
                and len(packed[-1]["text"]) + 1 + len(combined) <= max_chars
                and packed[-1]["section"] == section
            ):
                packed[-1]["text"] = packed[-1]["text"] + "\n" + combined
            elif combined:
                packed.append({"text": combined, "section": section})

    return packed


def chunk_pages(pages, filename, chunk_size=900, min_chars=80):
    """
    Structure-aware chunking with section-boundary hard breaks.
    """
    chunks = []
    base_name = Path(filename).name
    document_id = re.sub(
        r"[^a-z0-9_]",
        "",
        base_name.lower().replace(".pdf", "").replace("-", "_").replace(" ", "_"),
    )
    version = extract_version_from_filename(base_name)

    for page_item in pages:
        page_number = page_item["page"]
        page_text = clean_text(page_item["text"])
        starting_section = page_item.get("section", "General")
        blocks = split_into_blocks(page_text, starting_section=starting_section)
        packed = _pack_block_dicts(blocks, max_chars=chunk_size, min_chars=min_chars)

        for chunk_idx, item in enumerate(packed, 1):
            chunk_text = (item.get("text") or "").strip()
            if not chunk_text:
                continue
            chunk_id = f"{document_id}_p{page_number}_c{chunk_idx:02d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": base_name,
                "document_id": document_id,
                "page": page_number,
                "section": item.get("section", starting_section),
                "version": version,
            })

    return chunks


def generate_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


if __name__ == "__main__":
    pdf_dir = Path("data")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the data/ directory.")
        sys.exit(1)

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Ingesting: {pdf_path.name} ...")
        pages = extract_pages_from_pdf(str(pdf_path))
        chunks = chunk_pages(pages, pdf_path.name)
        print(f"  Generated {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"Total chunks generated across all documents: {len(all_chunks)}")

    # Sanity: attendance 75% should not be glued to seat-allocation table
    for c in all_chunks:
        if "75%" in c["text"] and "attendance" in c["text"].lower():
            head = c["text"][:80].replace("\n", " | ")
            print(f"  attendance/75% chunk: {c['chunk_id']} section={c['section']!r} head={head!r}")
            break

    if not all_chunks:
        print("No text chunks were generated.")
        sys.exit(1)

    embeddings = generate_embeddings(all_chunks)
    print(f"Embedding shape: {embeddings.shape}")

    with open("embeddings.pkl", "wb") as f:
        pickle.dump((all_chunks, embeddings), f)

    print("OK: Embeddings and metadata saved successfully to embeddings.pkl")
