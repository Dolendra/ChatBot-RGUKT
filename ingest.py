import pdfplumber
import re


def detect_section_header(text, current_section="General"):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return current_section

    # Check the first 3 lines of the page for headings
    for line in lines[:3]:
        # Match Roman numerals like "VI. SCHEME OF INSTRUCTION AND EXAMINATION"
        m = re.match(r"^([IVXLCDM]+)\.\s+([A-Z\s&,\-\(\)]+)$", line)
        if m:
            return m.group(2).strip()

        # Match other uppercase headings, e.g. "RULES AND REGULATIONS"
        if len(line) >= 5 and line.isupper() and re.match(r"^[A-Z\s&,\-\(\):\d]+$", line):
            if not re.match(r"^\d+$", line):
                return line

    return current_section


def extract_pages_from_pdf(pdf_path):
    pages = []
    current_section = "General"
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                current_section = detect_section_header(text, current_section)
                pages.append({
                    "page": i + 1,
                    "text": text,
                    "section": current_section,
                })
    return pages


_MAJOR_SECTION_RE = re.compile(
    r"^("
    r"[IVXLCDM]+\.\s+[A-Z][A-Z\s&,\-\(\)]{3,}"  # IV. RULES AND REGULATIONS...
    r"|[A-Z][A-Z\s&,\-\(\):\d]{8,}"  # long ALL-CAPS headings
    r")$"
)


def is_major_section_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) > 120:
        return False
    if not _MAJOR_SECTION_RE.match(s):
        return False
    # Avoid treating short all-caps noise / single tokens as sections
    letters = re.sub(r"[^A-Za-z]", "", s)
    return len(letters) >= 8


def section_title_from_heading(line: str) -> str | None:
    s = (line or "").strip()
    if not is_major_section_heading(s):
        return None
    m = re.match(r"^([IVXLCDM]+)\.\s+(.+)$", s)
    if m:
        return m.group(2).strip()
    return s


_HEADING_RE = re.compile(
    r"^("
    r"[IVXLCDM]+\.\s+.+"  # Roman numeral headings
    r"|\d+(\.\d+)*\.?\s+[A-Z].+"  # Numbered rules / subsections
    r"|[A-Z][A-Z\s&,\-\(\):\d]{4,}"  # ALL-CAPS headings
    r")$"
)
_BULLET_RE = re.compile(r"^([•●▪◦\-\*]|\d+[\.\)])\s+")
_TABLE_SEP_RE = re.compile(r"^[\-\|=_\s]{4,}$")


def _is_structural_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _BULLET_RE.match(s):
        return True
    if _HEADING_RE.match(s) and len(s) < 120:
        return True
    if _TABLE_SEP_RE.match(s):
        return True
    # Likely table / key-value row with multiple spaced columns
    if re.search(r"\S\s{2,}\S", s) and len(s.split()) >= 2:
        return True
    return False


def clean_text(text: str) -> str:
    """
    Clean PDF text while preserving academic structure:
    headings, bullets, numbered rules, paragraph boundaries, and table-like rows.
    Soft-wrapped paragraph lines are still merged.
    """
    if not text:
        return ""

    text = re.sub(r"\n--- Page \d+ ---\n", "\n", text)
    raw_lines = text.split("\n")
    out: list[str] = []
    buf: list[str] = []

    def flush_buf():
        if not buf:
            return
        paragraph = " ".join(part.strip() for part in buf if part.strip())
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            out.append(paragraph)
        buf.clear()

    for raw in raw_lines:
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_buf()
            if out and out[-1] != "":
                out.append("")
            continue

        if _is_structural_line(stripped):
            flush_buf()
            # Keep multi-space column gaps for table-like rows; normalize elsewhere lightly
            if re.search(r"\S\s{2,}\S", stripped):
                cleaned = re.sub(r"[ \t]{3,}", "  ", stripped)
            else:
                cleaned = re.sub(r"[ \t]+", " ", stripped)
            out.append(cleaned)
            continue

        # Soft-wrap continuation inside a paragraph
        buf.append(stripped)

    flush_buf()

    # Collapse 3+ blank lines to a single paragraph break
    cleaned_lines: list[str] = []
    blank_run = 0
    for line in out:
        if line == "":
            blank_run += 1
            if blank_run == 1:
                cleaned_lines.append("")
        else:
            blank_run = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def split_into_blocks(text: str, starting_section: str = "General") -> list[dict]:
    """
    Split cleaned page text into structural blocks for chunking.
    Each block: {"text": str, "section": str, "is_section_start": bool}
    Major section headings force a hard boundary (never merged with prior content).
    """
    if not text:
        return []

    blocks: list[dict] = []
    current: list[str] = []
    section = starting_section
    pending_section_start = False

    def flush():
        nonlocal pending_section_start
        if current:
            blocks.append({
                "text": "\n".join(current).strip(),
                "section": section,
                "is_section_start": pending_section_start,
            })
            current.clear()
            pending_section_start = False

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush()
            continue

        title = section_title_from_heading(stripped)
        if title:
            flush()
            section = title
            pending_section_start = True
            current.append(stripped)
            flush()
            continue

        starts_new = bool(_HEADING_RE.match(stripped) and len(stripped) < 120)
        if starts_new and current:
            flush()

        current.append(stripped)

        if _BULLET_RE.match(stripped) and len(stripped) < 400:
            flush()

    flush()
    return [b for b in blocks if b.get("text")]


if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/Academic_Regulations_Hand_Book.pdf"
    pages = extract_pages_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} non-empty pages from {pdf_path}")
    if pages:
        first_page = pages[0]
        print(f"First page section: {first_page['section']}")
        cleaned = clean_text(first_page["text"])
        print("\n--- First page preview (first 500 chars) ---\n")
        print(cleaned[:500])
        print(f"\nBlocks on first page: {len(split_into_blocks(cleaned))}")
        for b in split_into_blocks(cleaned)[:5]:
            print(f"  section={b.get('section')} start={b.get('is_section_start')} text={b.get('text','')[:80]!r}")
