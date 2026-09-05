import os
import re
import time
from typing import Any

from groq import Groq

from project_env import get_groq_api_key
from retrieve import retrieve_chunks
from verify import apply_verification, normalize_numbers_text

# --- Prompt / token controls (override via environment) ---
_RAG_CHARS = int(os.environ.get("RAG_CONTEXT_CHARS_PER_CHUNK", "1100"))
_HISTORY_MSG_CAP = int(os.environ.get("RAG_HISTORY_MAX_MESSAGES", "14"))
_HISTORY_CHARS = int(os.environ.get("RAG_HISTORY_CHARS_PER_MESSAGE", "450"))

_SYSTEM = """You have two inputs: (A) DOCUMENT EXCERPTS from indexed PDFs, (B) PRIOR CHAT turns.

Rules (follow strictly):
1) Facts about rules, policies, attendance, admissions, exams, syllabi: use ONLY (A). If (A) lacks the fact or retrieval is low confidence, reply exactly: I don't know from the documents. Cite each source at the end of the sentence or paragraph where it is used. Format citations as [Source n] (e.g. "Attendance must be 75% [Source 1]."). Only cite Source numbers that appear in (A).
2) Questions about THIS conversation ("we", "you said", "above", "our chat", "what I asked", "summarize the chat"): use ONLY (B). Do NOT describe random syllabus or course content from (A) as "what we discussed" unless the user clearly talked about that topic in (B).
3) Never invent citations or merge unrelated (A) into an answer about (B).
4) Be concise: short paragraphs, no preamble.
5) Do not invent page numbers, fees, percentages, or rule text that is not present in (A)."""

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

_FOLLOWUP_RE = re.compile(
    r"\b("
    r"what about|how about|and then|after that|before that|"
    r"same for|for that|for this|those|these|"
    r"it|them|they|that|this|the above|"
    r"exceptions?|requirements?|eligibility|documents?"
    r")\b",
    re.I,
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

    max_retries = int(os.environ.get("GROQ_MAX_RETRIES", "3"))
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                messages=messages,
                temperature=0.1,
                max_tokens=int(os.environ.get("GROQ_MAX_TOKENS", "900")),
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            name = type(e).__name__
            msg = str(e).lower()
            # Daily quota cannot be fixed by waiting a few seconds — fail immediately
            if "tokens per day" in msg or "tpd" in msg:
                raise
            retryable = "RateLimitError" in name or "rate_limit" in msg or "429" in msg
            if not retryable or attempt >= max_retries - 1:
                raise
            wait_s = 1.5 * (2 ** attempt)
            m = re.search(
                r"try again in (?:(\d+)m)?\s*([\d.]+)?\s*(ms|s)?",
                str(e),
                flags=re.I,
            )
            if m:
                minutes = float(m.group(1) or 0)
                val = float(m.group(2) or 0)
                unit = (m.group(3) or "s").lower()
                wait_s = minutes * 60.0
                if unit == "ms":
                    wait_s += val / 1000.0
                else:
                    wait_s += val
                wait_s = max(wait_s, 0.5) + 0.25
                wait_s = min(wait_s, 20.0)
            time.sleep(wait_s)
    raise last_err  # pragma: no cover


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


def _looks_like_followup(question: str) -> bool:
    q = question.strip()
    if len(q.split()) <= 8:
        return True
    return bool(_FOLLOWUP_RE.search(q))


def _rewrite_query(question: str, history: list[dict[str, Any]] | None) -> str:
    """
    Produce a standalone retrieval query.
    - Follow-ups: LLM rewrite using chat context
    - Standalone: keep original (lexical expansion happens inside retrieve.py)
    """
    q = question.strip()
    if not history or not _looks_like_followup(q):
        return q

    tail = _format_history(history[-6:] if history else None)
    if not tail:
        return q

    try:
        rewritten = _llm_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's latest message into a single standalone search query "
                        "for an academic regulations handbook. Preserve intent words "
                        "(minimum/required vs consequence/below/if). Preserve exact terms "
                        "(CGPA, B.Tech, attendance %, fees). "
                        "Output ONLY the rewritten query, no quotes or preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": f"PRIOR CHAT:\n{tail}\n\nLATEST MESSAGE:\n{q}",
                },
            ]
        )
        rewritten = (rewritten or "").strip().split("\n")[0].strip().strip('"')
        if 3 <= len(rewritten) <= 400:
            return rewritten
    except Exception:
        pass

    last_user = ""
    for m in reversed(history or []):
        if (m.get("role") or "").lower() == "user":
            last_user = (m.get("content") or "").strip()
            break
    if last_user and last_user.lower() != q.lower():
        return f"{last_user} — {q}"[:500]
    return q


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


def _normalize_citation_markers(answer: str) -> str:
    """
    Canonicalize common citation spellings to [Source n] before validation.
    """
    if not answer:
        return answer
    out = answer
    # 【Source 1】 or [Source 1] already
    out = re.sub(r"[【\[]\s*Source\s+(\d+)\s*[】\]]", r"[Source \1]", out, flags=re.I)
    # (Source 1)
    out = re.sub(r"\(\s*Source\s+(\d+)\s*\)", r"[Source \1]", out, flags=re.I)
    # Source 1 / Source1 as standalone tokens (avoid URLs etc.)
    out = re.sub(r"(?<!\[)\bSource\s*(\d+)\b(?!\])", r"[Source \1]", out, flags=re.I)
    return out


def _validate_citations(answer: str, n_sources: int) -> str:
    """
    Normalize citation markers, then keep only those that map to retrieved sources.
    """
    if not answer:
        return answer

    answer = _normalize_citation_markers(answer)

    def repl(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= n_sources:
            return f"[Source {n}]"
        return ""

    cleaned = re.sub(r"\[Source\s+(\d+)\]", repl, answer, flags=re.I)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
    return cleaned.strip()


def _doc_block_and_chunks(question: str, top_k: int, history: list[dict[str, Any]] | None):
    empty_metrics = {
        "embedding_ms": 0.0,
        "faiss_ms": 0.0,
        "bm25_ms": 0.0,
        "rrf_ms": 0.0,
        "rerank_ms": 0.0,
        "total_retrieve_ms": 0.0,
        "top_score": 0.0,
        "low_confidence": False,
        "search_query": question.strip(),
    }
    if _skip_docs_for_meta(question, history):
        return "(No document lookup for this turn — answer from prior chat only.)", [], empty_metrics

    search_query = _rewrite_query(question, history)
    threshold = float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "-5.0"))

    chunks, metrics = retrieve_chunks(
        search_query, top_k=top_k, return_metrics=True, threshold=threshold
    )
    metrics = {**metrics, "search_query": search_query}

    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        body = _trim_doc(chunk.get("text", ""), _RAG_CHARS)
        p = chunk.get("page", "—")
        src = chunk.get("source", "Unknown PDF")
        sec = chunk.get("section", "General")
        parts.append(f"[Source {i} | File: {src} | Page: {p} | Section: {sec}]\n{body}")

    return "\n\n".join(parts), chunks, metrics


def run_rag(question: str, top_k: int = 3, history: list[dict[str, Any]] | None = None):
    """
    Returns (answer_text, sources, metrics). `history` is prior turns only.

    Pipeline: retrieve → draft LLM → answer verifier (PASS/REVISE/FAIL) → citation validation.
    """
    t_start = time.time()
    doc_block, chunks, retrieve_metrics = _doc_block_and_chunks(question, top_k, history)

    # 1. Abstain check if low confidence
    if retrieve_metrics.get("low_confidence", False):
        t_total = time.time() - t_start
        generation_metrics = {
            "generation_ms": 0.0,
            "verify_ms": 0.0,
            "total_ms": round(t_total * 1000, 2),
            "verification": {"verification_enabled": False, "decision": "SKIP_LOW_CONFIDENCE"},
        }
        combined_metrics = {**retrieve_metrics, **generation_metrics}
        return (
            "I couldn't find any relevant information in the RGUKT documents.",
            [],
            combined_metrics,
        )

    # 2. Draft LLM call
    t_llm_start = time.time()
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

    draft = _llm_chat(messages)
    t_draft = time.time() - t_llm_start

    # 3. Answer verification (question + evidence + draft only — no chat history)
    t_verify_start = time.time()
    if chunks and not _skip_docs_for_meta(question, history):
        answer, verify_info = apply_verification(
            question=question.strip(),
            evidence_block=doc_block,
            draft=draft,
            n_sources=len(chunks),
            llm_chat=_llm_chat,
        )
    else:
        answer, verify_info = draft, {"verification_enabled": False, "decision": "SKIP_META"}
    t_verify = time.time() - t_verify_start

    # 4. Citation validation
    answer = _validate_citations(answer, len(chunks))
    # Normalize unicode number spacing in the final user-facing answer
    answer = normalize_numbers_text(answer) if answer else answer

    t_total = time.time() - t_start
    generation_metrics = {
        "generation_ms": round(t_draft * 1000, 2),
        "verify_ms": round(t_verify * 1000, 2),
        "total_ms": round(t_total * 1000, 2),
        "verification": verify_info,
    }
    combined_metrics = {**retrieve_metrics, **generation_metrics}

    sources = [
        {
            "id": i + 1,
            "chunk_id": c.get("chunk_id", "unknown"),
            "source": c.get("source", "Academic_Regulations_Hand_Book.pdf"),
            "document_id": c.get("document_id", "academic_regulations_hand_book"),
            "page": c.get("page", "—"),
            "section": c.get("section", "General"),
            "version": c.get("version", "—"),
            "score": c.get("score"),
            "snippet": _trim_doc(c.get("text", ""), 520),
        }
        for i, c in enumerate(chunks)
    ]

    return answer, sources, combined_metrics


def rag_answer(question: str, top_k: int = 3):
    answer, _, _ = run_rag(question, top_k=top_k, history=None)
    return answer


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        answer, sources, metrics = run_rag(query)
        print("\nAnswer:\n", answer)
        print("\nMetrics:\n", metrics)
        print("\nSources:\n", sources)
