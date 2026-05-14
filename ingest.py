import pdfplumber
import re


def extract_pages_from_pdf(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append((i + 1, text))
    return pages


def clean_text(text):
    # 1. Remove page markers
    text = re.sub(r"\n--- Page \d+ ---\n", "\n", text)

    # 2. Merge broken lines inside paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 3. Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/Academic_Regulations_Hand_Book.pdf"
    pages = extract_pages_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} non-empty pages from {pdf_path}")
    if pages:
        first_page_text = clean_text(pages[0][1])
        print("\n--- First page preview (first 500 chars) ---\n")
        print(first_page_text[:500])

