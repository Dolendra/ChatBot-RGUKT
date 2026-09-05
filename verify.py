"""
Answer verification for grounded RAG responses.

Given QUESTION + EVIDENCE + DRAFT (no chat history), decide:
  PASS  → keep draft
  REVISE → constrained rewrite from evidence only
  FAIL  → refuse
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable


_ABSTAIN = "I don't know from the documents."

_VERIFY_SYSTEM = """You are an answer verification system for a university academic regulations chatbot.

Your job is NOT to answer the user's question.

Given:
1. User question
2. Retrieved document evidence
3. Draft answer

Check whether every factual claim in the draft is supported by the retrieved evidence.

Check:
- factual claim support
- numerical consistency (percentages, fees, marks, CGPA, thresholds)
- dates, conditions, and exceptions
- completeness of important rules and consequences present in the evidence that directly answer the question
- citation-to-claim grounding ([Source n] must support the preceding claim)
- contradictions between the draft and evidence

Do not use outside knowledge.

For each claim, classify it as:
SUPPORTED
UNSUPPORTED
CONTRADICTED

If the answer is fully supported and adequately complete for the question:
decision = PASS

If it can be corrected using ONLY the supplied evidence:
decision = REVISE

If the evidence is insufficient or contradictory:
decision = FAIL

Return structured JSON only with this schema:
{
  "decision": "PASS" | "REVISE" | "FAIL",
  "claims": [
    {"claim": "...", "status": "SUPPORTED|UNSUPPORTED|CONTRADICTED", "sources": [1]}
  ],
  "missing_points": ["..."],
  "number_issues": ["..."],
  "citation_issues": ["..."],
  "notes": "short rationale"
}
"""

_REVISE_SYSTEM = """You revise a draft answer for a university academic regulations chatbot.

Rules:
1) Use ONLY the supplied DOCUMENT EXCERPTS.
2) Fix unsupported, contradicted, incomplete, or numerically wrong claims.
3) Include important consequences/conditions from the evidence that answer the question.
4) Cite with [Source n] only for sources that support the claim.
5) Be concise. No preamble.
6) If the evidence still cannot answer, reply exactly: I don't know from the documents.
"""


def normalize_numbers_text(text: str) -> str:
    """Normalize unicode spaces/dashes around numbers and percentages for comparison."""
    if not text:
        return ""
    t = text
    # unicode spaces → normal space
    t = re.sub(r"[\u00a0\u202f\u2007\u2009\u200a\u200b]", " ", t)
    # normalize percent forms: 75 % / 75％ → 75%
    t = re.sub(r"(\d+(?:\.\d+)?)\s*[%％]", r"\1%", t)
    # collapse horizontal whitespace only (preserve newlines)
    t = re.sub(r"[^\S\n]+", " ", t)
    return t.strip()


def extract_number_tokens(text: str) -> set[str]:
    t = normalize_numbers_text(text).lower()
    return set(re.findall(r"\d[\d,]*(?:\.\d+)?%?", t))


def _extract_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    text = raw.strip()
    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Best-effort: first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _heuristic_number_conflict(question: str, evidence: str, draft: str) -> list[str]:
    """
    Detect numbers present in the draft that appear in neither evidence nor question.
    Also flag when draft uses a number that conflicts with a clear evidence percentage
    for the same topic (lightweight, not full NLI).
    """
    issues: list[str] = []
    draft_nums = extract_number_tokens(draft)
    evid_nums = extract_number_tokens(evidence)
    q_nums = extract_number_tokens(question)
    allowed = evid_nums | q_nums
    for n in draft_nums:
        # Ignore trivial single digits that are often list markers
        if re.fullmatch(r"\d", n):
            continue
        if n not in allowed:
            issues.append(f"Draft number {n} not found in evidence or question")
    return issues


def verify_draft(
    *,
    question: str,
    evidence_block: str,
    draft: str,
    llm_chat: Callable[[list[dict[str, str]]], str],
    n_sources: int,
) -> dict[str, Any]:
    """
    Run claim-level verification. Returns a normalized result dict.
    """
    draft = (draft or "").strip()
    if not draft:
        return {
            "decision": "FAIL",
            "claims": [],
            "missing_points": ["Empty draft"],
            "number_issues": [],
            "citation_issues": [],
            "notes": "empty draft",
            "raw": None,
        }

    # Skip heavy verification if already abstaining
    if draft.lower().startswith("i don't know") or "couldn't find" in draft.lower():
        return {
            "decision": "PASS",
            "claims": [],
            "missing_points": [],
            "number_issues": [],
            "citation_issues": [],
            "notes": "draft already abstains",
            "raw": None,
        }

    heuristic_nums = _heuristic_number_conflict(question, evidence_block, draft)

    user_payload = (
        f"QUESTION:\n{question.strip()}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"There are {n_sources} numbered sources in EVIDENCE."
    )

    raw = llm_chat(
        [
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user", "content": user_payload},
        ]
    )
    parsed = _extract_json(raw) or {}

    decision = str(parsed.get("decision", "REVISE")).upper().strip()
    if decision not in ("PASS", "REVISE", "FAIL"):
        decision = "REVISE"

    claims = parsed.get("claims") if isinstance(parsed.get("claims"), list) else []
    unsupported = [
        c for c in claims
        if isinstance(c, dict) and str(c.get("status", "")).upper() in ("UNSUPPORTED", "CONTRADICTED")
    ]
    number_issues = list(parsed.get("number_issues") or [])
    number_issues.extend(heuristic_nums)
    missing_points = list(parsed.get("missing_points") or [])
    citation_issues = list(parsed.get("citation_issues") or [])

    # Escalate PASS → REVISE if heuristics / claims disagree
    if decision == "PASS" and (unsupported or number_issues or missing_points):
        decision = "REVISE"
    if decision == "PASS" and any(
        str(c.get("status", "")).upper() == "CONTRADICTED" for c in claims if isinstance(c, dict)
    ):
        decision = "REVISE"

    # If many contradictions and no supported claims → FAIL
    contradicted = [
        c for c in claims
        if isinstance(c, dict) and str(c.get("status", "")).upper() == "CONTRADICTED"
    ]
    supported = [
        c for c in claims
        if isinstance(c, dict) and str(c.get("status", "")).upper() == "SUPPORTED"
    ]
    if contradicted and not supported and not missing_points:
        decision = "FAIL"

    return {
        "decision": decision,
        "claims": claims,
        "missing_points": missing_points,
        "number_issues": number_issues,
        "citation_issues": citation_issues,
        "notes": parsed.get("notes", ""),
        "raw": raw,
    }


def revise_answer(
    *,
    question: str,
    evidence_block: str,
    draft: str,
    verification: dict[str, Any],
    llm_chat: Callable[[list[dict[str, str]]], str],
) -> str:
    """Constrained revision using only evidence + verifier findings."""
    issues = []
    if verification.get("missing_points"):
        issues.append("Missing points:\n- " + "\n- ".join(map(str, verification["missing_points"])))
    if verification.get("number_issues"):
        issues.append("Number issues:\n- " + "\n- ".join(map(str, verification["number_issues"])))
    if verification.get("citation_issues"):
        issues.append("Citation issues:\n- " + "\n- ".join(map(str, verification["citation_issues"])))
    unsupported = [
        c.get("claim")
        for c in (verification.get("claims") or [])
        if isinstance(c, dict) and str(c.get("status", "")).upper() in ("UNSUPPORTED", "CONTRADICTED")
    ]
    if unsupported:
        issues.append("Unsupported/contradicted claims to remove or fix:\n- " + "\n- ".join(map(str, unsupported)))

    payload = (
        f"QUESTION:\n{question.strip()}\n\n"
        f"DOCUMENT EXCERPTS:\n{evidence_block}\n\n"
        f"DRAFT ANSWER:\n{draft}\n\n"
        f"VERIFIER FINDINGS:\n" + ("\n\n".join(issues) if issues else "(revise for grounding and completeness)")
        + "\n\nReturn only the revised answer."
    )
    return llm_chat(
        [
            {"role": "system", "content": _REVISE_SYSTEM},
            {"role": "user", "content": payload},
        ]
    ).strip()


def apply_verification(
    *,
    question: str,
    evidence_block: str,
    draft: str,
    n_sources: int,
    llm_chat: Callable[[list[dict[str, str]]], str],
    enabled: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Returns (final_answer, verification_metrics).
    """
    if enabled is None:
        enabled = os.environ.get("RAG_VERIFY", "1").strip() not in ("0", "false", "False")

    if not enabled or n_sources <= 0:
        return draft, {"verification_enabled": False, "decision": "SKIP"}

    result = verify_draft(
        question=question,
        evidence_block=evidence_block,
        draft=draft,
        llm_chat=llm_chat,
        n_sources=n_sources,
    )
    decision = result["decision"]
    metrics = {
        "verification_enabled": True,
        "decision": decision,
        "unsupported_claims": sum(
            1
            for c in result.get("claims") or []
            if isinstance(c, dict) and str(c.get("status", "")).upper() in ("UNSUPPORTED", "CONTRADICTED")
        ),
        "missing_points": len(result.get("missing_points") or []),
        "number_issues": len(result.get("number_issues") or []),
        "citation_issues": len(result.get("citation_issues") or []),
    }

    if decision == "PASS":
        return draft, metrics
    if decision == "FAIL":
        return _ABSTAIN, metrics

    # REVISE
    revised = revise_answer(
        question=question,
        evidence_block=evidence_block,
        draft=draft,
        verification=result,
        llm_chat=llm_chat,
    )
    if not revised:
        return _ABSTAIN, {**metrics, "decision": "FAIL"}
    return revised, metrics
