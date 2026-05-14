"""Load `.env` from the repository root and read cleaned secrets (works regardless of process cwd)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
_DOTENV_DONE = False


def ensure_dotenv() -> None:
    global _DOTENV_DONE
    if _DOTENV_DONE:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    _DOTENV_DONE = True


def normalize_secret(value: str) -> str:
    """Strip whitespace, UTF-8 BOM, and optional surrounding quotes (common .env paste issues)."""
    if not value:
        return ""
    s = value.strip()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def get_groq_api_key() -> str:
    ensure_dotenv()
    k = normalize_secret(os.environ.get("GROQ_API_KEY", ""))
    # Groq keys are a single token; remove accidental spaces / line breaks inside the value
    if k.startswith("gsk_"):
        k = "gsk_" + "".join(k[4:].split())
    return k


def groq_key_configured() -> bool:
    return bool(get_groq_api_key())


def _rag_pdf_path() -> Path:
    ensure_dotenv()
    raw = (os.environ.get("RAG_PDF_PATH") or "data/Academic_Regulations_Hand_Book.pdf").strip()
    return Path(raw)


def document_title_from_path(pdf: Path) -> str:
    """Readable title from PDF filename (e.g. Academic_Regulations_Hand_Book.pdf)."""
    stem = re.sub(r"[_\-]+", " ", pdf.stem).strip()
    if not stem:
        return "Indexed documents"
    return stem.title()


def get_app_title() -> str:
    """UI title: APP_TITLE if set, else derived from RAG_PDF_PATH filename."""
    ensure_dotenv()
    custom = normalize_secret(os.environ.get("APP_TITLE", ""))
    if custom:
        return custom
    return f"{document_title_from_path(_rag_pdf_path())} · Assistant"


def get_app_subtitle() -> str:
    """Short subtitle under the title."""
    ensure_dotenv()
    custom = normalize_secret(os.environ.get("APP_SUBTITLE", ""))
    if custom:
        return custom
    fname = _rag_pdf_path().name
    return f"Ask about {fname} — answers cite retrieved passages."
