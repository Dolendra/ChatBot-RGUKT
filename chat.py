import os
import re
from typing import Any

from groq import Groq

from project_env import get_groq_api_key
from retrieve import retrieve_chunks

# --- Prompt / token controls (override via environment) ---
_RAG_CHARS = int(os.environ.get("RAG_CONTEXT_CHARS_PER_CHUNK", "1100"))
_HISTORY_MSG_CAP = int(os.environ.get("RAG_HISTORY_MAX_MESSAGES", "14"))
_HISTORY_CHARS = int(os.environ.get("RAG_HISTORY_CHARS_PER_MESSAGE", "450"))

_SYSTEM = """You have two inputs: (A) DOCUMENT EXCERPTS from indexed PDFs, (B) PRIOR CHAT turns.

Rules (follow strictly):
1) Facts about rules, policies, attendance, admissions, exams, syllabi: use ONLY (A). If (A) lacks the fact, reply exactly: I don't know from the documents. Cite as (Source n, Page p) only when p is a number; if page is unknown use (Source n).
2) Questions about THIS conversation ("we", "you said", "above", "our chat", "what I asked", "summarize the chat"): use ONLY (B). Do NOT describe random syllabus or course content from (A) as "what we discussed" unless the user clearly talked about that topic in (B).
3) Never invent citations or merge unrelated (A) into an answer about (B).
4) Be concise: short paragraphs, no preamble."""

_META_NEEDLES = (
    "this chat",
    "our chat",
    "the chat",
    "what we discussed",
    "you said",
    "your answer",
    "your reply",
    "previous answer",
    "above answer",
    "above message",
    "earlier you",
    "what did i ask",
    "what was my question",
    "what was the question",
    "above chat",
    "recap the",
    "summarize the chat",
    "summarise the chat",
    "what is the chat",
    "what's the chat",
    "what we are discussing",
    "what are we discussing",
)


def _groq_client():
    key = get_groq_api_key()
    if not key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Copy .env.example to .env in the project folder, set your key "
            "on one line (no spaces around '='), or export GROQ_API_KEY in your shell."
        )
    return Groq(api_key=key)


client = None


def _llm_chat(messages: list[dict[str, str]]) -> str:
    global client
    if client is None:
        client = _groq_client()
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
        temperature=0.1,
        max_tokens=int(os.environ.get("GROQ_MAX_TOKENS", "900")),
    )
    return (response.choices[0].message.content or "").strip()


def _format_history(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for m in history[-_HISTORY_MSG_CAP:]:
        role = (m.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        raw = (m.get("content") or "").strip().replace("\n", " ")
        if not raw:
            continue
        if len(raw) > _HISTORY_CHARS:
            raw = raw[: _HISTORY_CHARS - 1] + "…"
        label = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"{label}: {raw}")
    return "\n".join(lines)


def _retrieval_query(question: str, history: list[dict[str, Any]] | None) -> str:
    """Bias embeddings toward follow-ups by appending a short recent-chat tail."""
    q = question.strip()
    tail = _format_history(history)
    if not tail:
        return q
    merged = f"{q}\n\nRecent chat (for context):\n{tail}"
    return merged[:2500]


def _skip_docs_for_meta(question: str, history: list[dict[str, Any]] | None) -> bool:
    """Avoid irrelevant PDF hits for clear conversation-meta questions."""
    if not history:
        return False
    ql = question.lower()
    if any(n in ql for n in _META_NEEDLES):
        return True
    if re.search(r"\b(what|who)\s+.+\s+above\b", ql) and "chat" in ql:
        return True
    return False


def _trim_doc(text: str, limit: int) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "…"


def _doc_block_and_chunks(question: str, top_k: int, history: list[dict[str, Any]] | None):
    if _skip_docs_for_meta(question, history):
        return "(No document lookup for this turn — answer from prior chat only.)", []
    q = _retrieval_query(question, history)
    chunks = retrieve_chunks(q, top_k=top_k)
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        body = _trim_doc(chunk.get("text", ""), _RAG_CHARS)
        p = chunk.get("page", "—")
        parts.append(f"[Source {i} | Page {p}]\n{body}")
    return "\n\n".join(parts), chunks


def run_rag(question: str, top_k: int = 3, history: list[dict[str, Any]] | None = None):
    """
    Returns (answer_text, sources). `history` is prior turns only (excludes current message).
    """
    doc_block, chunks = _doc_block_and_chunks(question, top_k, history)
    hist_block = _format_history(history)

    user_parts = []
    if hist_block:
        user_parts.append("PRIOR CHAT (for questions about this conversation):\n" + hist_block)
    user_parts.append("DOCUMENT EXCERPTS (for policy / handbook facts only):\n" + doc_block)
    user_parts.append("CURRENT USER MESSAGE:\n" + question.strip())
    user_payload = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_payload},
    ]
    answer = _llm_chat(messages)

    sources = [
        {
            "id": i + 1,
            "page": c.get("page", "—"),
            "snippet": _trim_doc(c.get("text", ""), 520),
        }
        for i, c in enumerate(chunks)
    ]
    return answer, sources


def rag_answer(question: str, top_k: int = 3):
    answer, _ = run_rag(question, top_k=top_k, history=None)
    return answer


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        answer = rag_answer(query)
        print("\nAnswer:\n", answer)
