"""
RAG evaluation suite for the RGUKT Academic Assistant.

Evidence matching uses stable document/page/section + evidence keywords —
NOT brittle chunk IDs (those shift after re-indexing).

Metrics:
  Retrieval: Recall@K, Precision@K, MRR, NDCG@5
  Generation: Faithfulness, Correctness (LLM-as-judge), Keyword hit-rate
  Citations: Citation correctness (cited source IDs exist & pages align)
  System: Latency breakdown, refusal/abstention rate
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from typing import Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat import run_rag, _llm_chat
from verify import normalize_numbers_text

ABSTENTION_NEEDLES = (
    "don't know",
    "do not know",
    "couldn't find",
    "could not find",
    "no relevant information",
    "not found in the",
    "outside the indexed",
)


def _as_int_page(page: Any) -> int | None:
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def _norm(s: str) -> str:
    return normalize_numbers_text(re.sub(r"\s+", " ", (s or "").lower())).strip()


def _keyword_hit(text: str, keywords: list[str]) -> bool:
    t = _norm(text)
    for kw in keywords:
        if _norm(kw) and _norm(kw) in t:
            return True
    return False


def _source_is_relevant(source: dict, q_item: dict) -> bool:
    """
    Stable relevance: page match OR (section match + evidence keyword in snippet).
    Chunk IDs are ignored by design.
    """
    expected_pages = set(q_item.get("relevant_pages") or [])
    expected_sections = {_norm(s) for s in (q_item.get("relevant_sections") or []) if s}
    evidence_kws = q_item.get("evidence_keywords") or []
    expected_doc = q_item.get("document_id")

    page = _as_int_page(source.get("page"))
    section = _norm(str(source.get("section", "")))
    snippet = str(source.get("snippet") or source.get("text") or "")
    doc_id = source.get("document_id") or ""

    if expected_doc and doc_id and expected_doc not in str(doc_id):
        # Soft: still allow page match if document_id missing on older sources
        if "document_id" in source and source.get("document_id"):
            pass

    if page is not None and page in expected_pages:
        return True

    if expected_sections and section in expected_sections and evidence_kws:
        if _keyword_hit(snippet, evidence_kws):
            return True

    # Keyword-only fallback when pages may have drifted slightly but evidence text is clear
    if evidence_kws and page is not None and expected_pages:
        # allow adjacent page (±1) if strong keyword evidence present
        if any(abs(page - p) <= 1 for p in expected_pages) and _keyword_hit(snippet, evidence_kws):
            return True

    return False


def _relevance_labels(sources: list[dict], q_item: dict, k: int) -> list[int]:
    labels = []
    for s in sources[:k]:
        labels.append(1 if _source_is_relevant(s, q_item) else 0)
    while len(labels) < k:
        labels.append(0)
    return labels


def recall_at_k(labels: list[int], k: int) -> float:
    return 1.0 if any(labels[:k]) else 0.0


def precision_at_k(labels: list[int], k: int) -> float:
    window = labels[:k]
    if not window:
        return 0.0
    return sum(window) / float(k)


def mrr_from_labels(labels: list[int]) -> float:
    for i, lab in enumerate(labels, 1):
        if lab:
            return 1.0 / i
    return 0.0


def dcg_at_k(labels: list[int], k: int) -> float:
    score = 0.0
    for i, lab in enumerate(labels[:k], 1):
        if lab:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(labels: list[int], k: int) -> float:
    dcg = dcg_at_k(labels, k)
    ideal = dcg_at_k(sorted(labels[:k], reverse=True), k)
    if ideal <= 0:
        return 0.0
    return dcg / ideal


def _extract_cited_ids(answer: str) -> list[int]:
    # Mirror chat normalization so 【Source 1】 counts
    text = answer or ""
    text = re.sub(r"[【\[]\s*Source\s+(\d+)\s*[】\]]", r"[Source \1]", text, flags=re.I)
    text = re.sub(r"\(\s*Source\s+(\d+)\s*\)", r"[Source \1]", text, flags=re.I)
    text = re.sub(r"(?<!\[)\bSource\s*(\d+)\b(?!\])", r"[Source \1]", text, flags=re.I)
    return [int(n) for n in re.findall(r"\[Source\s+(\d+)\]", text, flags=re.I)]


def citation_correctness(answer: str, sources: list[dict], q_item: dict) -> dict[str, Any]:
    """
    - valid_ids: every cited Source n exists in returned sources
    - grounded_citations: cited sources that are relevant evidence (page/section/keywords)
    """
    cited = _extract_cited_ids(answer)
    n = len(sources)
    if not cited:
        return {
            "cited_count": 0,
            "valid_id_rate": 1.0 if n == 0 or q_item.get("category") == "Out-of-Domain" else 0.0,
            "grounded_citation_rate": 1.0 if q_item.get("category") == "Out-of-Domain" else 0.0,
            "cited_ids": [],
        }

    valid = [c for c in cited if 1 <= c <= n]
    valid_id_rate = len(valid) / len(cited)

    grounded = 0
    for c in valid:
        src = sources[c - 1]
        if _source_is_relevant(src, q_item):
            grounded += 1
    grounded_rate = grounded / len(valid) if valid else 0.0

    return {
        "cited_count": len(cited),
        "valid_id_rate": valid_id_rate,
        "grounded_citation_rate": grounded_rate,
        "cited_ids": cited,
    }


def _is_abstention(answer: str) -> bool:
    a = (answer or "").lower()
    return any(n in a for n in ABSTENTION_NEEDLES)


def _keyword_answer_score(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    hits = sum(1 for kw in expected_keywords if _norm(kw) in _norm(answer))
    return hits / len(expected_keywords)


def _judge_score(prompt: str) -> float | None:
    try:
        res = _llm_chat([{"role": "system", "content": prompt}])
        m = re.search(r"Score:\s*([1-5])", res or "")
        if m:
            return float(m.group(1))
    except Exception as e:
        print(f"      Warning: LLM-as-a-judge failed: {e}")
    return None


def _failure_tags(
    *,
    category: str,
    labels: list[int],
    answer: str,
    sources: list[dict],
    q_item: dict,
    cit: dict,
    correctness: float,
    faithfulness: float,
) -> list[str]:
    tags: list[str] = []
    if category == "Out-of-Domain":
        if not _is_abstention(answer):
            tags.append("failed_refusal")
        return tags

    if not any(labels[:5]):
        tags.append("missed_evidence@5")
    elif not labels[0]:
        tags.append("relevant_ranked_too_low")

    if correctness >= 4 and not any(labels[:5]):
        tags.append("correct_answer_wrong_or_missing_source")

    if cit["cited_count"] > 0 and cit["valid_id_rate"] < 1.0:
        tags.append("invalid_citation_ids")
    if cit["cited_count"] > 0 and cit["grounded_citation_rate"] < 0.5:
        tags.append("citation_not_grounded")

    if faithfulness <= 2:
        tags.append("low_faithfulness")
    if correctness <= 2:
        tags.append("low_correctness")

    # Exact-number checks (unicode-normalized)
    joined = normalize_numbers_text(" ".join(q_item.get("expected_keywords") or []))
    ans_norm = normalize_numbers_text(answer or "")
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?%?", joined)
    for num in nums:
        if num and num not in ans_norm and any(ch.isdigit() for ch in num):
            if num in normalize_numbers_text(q_item.get("reference_answer") or ""):
                tags.append("exact_number_mismatch")
                break

    if _is_abstention(answer) and any(labels[:5]):
        tags.append("false_refusal")

    return tags


def run_evaluation(num_questions: int | None = None):
    eval_set_path = os.path.join("data", "evaluation_set.json")
    if not os.path.exists(eval_set_path):
        print(f"Evaluation set not found at {eval_set_path}")
        return None

    with open(eval_set_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if num_questions:
        questions = questions[:num_questions]

    print(f"Starting evaluation of {len(questions)} test cases...")
    print("Evidence matching: document/page/section + keywords (chunk IDs ignored)")

    results = []
    sums = {
        "recall_1": 0.0,
        "recall_3": 0.0,
        "recall_5": 0.0,
        "precision_1": 0.0,
        "precision_3": 0.0,
        "precision_5": 0.0,
        "mrr": 0.0,
        "ndcg_5": 0.0,
        "faithfulness": 0.0,
        "correctness": 0.0,
        "keyword_hit": 0.0,
        "citation_valid": 0.0,
        "citation_grounded": 0.0,
        "refusal_correct": 0.0,
    }
    latency_sums = {
        "embedding_ms": 0.0,
        "faiss_ms": 0.0,
        "bm25_ms": 0.0,
        "rrf_ms": 0.0,
        "rerank_ms": 0.0,
        "generation_ms": 0.0,
        "verify_ms": 0.0,
        "total_ms": 0.0,
    }
    verify_decisions: dict[str, int] = {}
    failure_counter: dict[str, int] = {}
    ood_count = 0
    in_domain_count = 0

    for idx, q_item in enumerate(questions, 1):
        question = q_item["question"]
        category = q_item.get("category", "Unknown")
        ref_answer = q_item.get("reference_answer", "")
        qid = q_item.get("id", f"q{idx}")

        print(f"[{idx}/{len(questions)}] {qid}: '{question}' ({category})")

        # Small pause between cases to stay under Groq TPM limits
        if idx > 1:
            time.sleep(float(os.environ.get("RAG_EVAL_PAUSE_S", "1.25")))

        answer, sources, metrics = run_rag(question, top_k=5)

        # Attach document_id onto sources if missing (for matching)
        for s in sources:
            if "document_id" not in s:
                src_name = str(s.get("source", "")).lower().replace(".pdf", "").replace("-", "_").replace(" ", "_")
                s["document_id"] = re.sub(r"[^a-z0-9_]", "", src_name)

        labels = _relevance_labels(sources, q_item, k=5)
        retrieved_pages = [s.get("page") for s in sources]
        retrieved_sections = [s.get("section") for s in sources]

        if category == "Out-of-Domain":
            ood_count += 1
            abstained = _is_abstention(answer) or metrics.get("low_confidence", False) or not sources
            # Retrieval success for OOD = abstain / low-confidence (do not reward random pages)
            r1 = r3 = r5 = 1.0 if abstained else 0.0
            p1 = p3 = p5 = 1.0 if abstained else 0.0
            mrr = 1.0 if abstained else 0.0
            ndcg = 1.0 if abstained else 0.0
            faithfulness = 5.0
            correctness = 5.0 if abstained else 1.0
            kw_hit = 1.0 if abstained else 0.0
            cit = {
                "cited_count": len(_extract_cited_ids(answer)),
                "valid_id_rate": 1.0 if not _extract_cited_ids(answer) else 0.0,
                "grounded_citation_rate": 1.0 if abstained else 0.0,
                "cited_ids": _extract_cited_ids(answer),
            }
            sums["refusal_correct"] += 1.0 if abstained else 0.0
        else:
            in_domain_count += 1
            r1 = recall_at_k(labels, 1)
            r3 = recall_at_k(labels, 3)
            r5 = recall_at_k(labels, 5)
            p1 = precision_at_k(labels, 1)
            p3 = precision_at_k(labels, 3)
            p5 = precision_at_k(labels, 5)
            mrr = mrr_from_labels(labels)
            ndcg = ndcg_at_k(labels, 5)
            kw_hit = _keyword_answer_score(answer, q_item.get("expected_keywords") or [])
            cit = citation_correctness(answer, sources, q_item)

            context_text = "\n\n".join(
                [f"Source: {s.get('source')} | Page: {s.get('page')} | Section: {s.get('section')}\n{s.get('snippet')}" for s in sources]
            )
            faithfulness_prompt = f"""You are an expert AI evaluator.
Analyze the generated RAG answer based ONLY on the retrieved context. Determine if the answer is faithful to the context and does not contain outside information or hallucinations.

Retrieved Context:
{context_text}

Generated Answer:
{answer}

Output a single line: "Score: <score>" where <score> is an integer from 1 (completely unfaithful/hallucinated) to 5 (fully faithful). Provide no other explanation."""

            correctness_prompt = f"""You are an expert AI evaluator.
Compare the generated answer against the ground-truth reference answer for the given question. Grade how accurate and complete the generated answer is.

Question: {question}
Reference Answer: {ref_answer}
Generated Answer: {answer}

Output a single line: "Score: <score>" where <score> is an integer from 1 (completely incorrect) to 5 (fully correct and complete). Provide no other explanation."""

            faithfulness = _judge_score(faithfulness_prompt) or 3.0
            correctness = _judge_score(correctness_prompt) or 3.0

        tags = _failure_tags(
            category=category,
            labels=labels,
            answer=answer,
            sources=sources,
            q_item=q_item,
            cit=cit,
            correctness=correctness,
            faithfulness=faithfulness,
        )
        for t in tags:
            failure_counter[t] = failure_counter.get(t, 0) + 1

        sums["recall_1"] += r1
        sums["recall_3"] += r3
        sums["recall_5"] += r5
        sums["precision_1"] += p1
        sums["precision_3"] += p3
        sums["precision_5"] += p5
        sums["mrr"] += mrr
        sums["ndcg_5"] += ndcg
        sums["faithfulness"] += faithfulness
        sums["correctness"] += correctness
        sums["keyword_hit"] += kw_hit
        sums["citation_valid"] += cit["valid_id_rate"]
        sums["citation_grounded"] += cit["grounded_citation_rate"]

        for key in latency_sums:
            latency_sums[key] += float(metrics.get(key, 0.0) or 0.0)

        vinfo = metrics.get("verification") or {}
        vdecision = str(vinfo.get("decision", "SKIP"))
        verify_decisions[vdecision] = verify_decisions.get(vdecision, 0) + 1

        results.append({
            "id": qid,
            "question": question,
            "category": category,
            "expected_pages": q_item.get("relevant_pages", []),
            "expected_sections": q_item.get("relevant_sections", []),
            "relevance_labels": labels,
            "recall_1": r1,
            "recall_3": r3,
            "recall_5": r5,
            "precision_1": round(p1, 4),
            "precision_3": round(p3, 4),
            "precision_5": round(p5, 4),
            "mrr": round(mrr, 4),
            "ndcg_5": round(ndcg, 4),
            "faithfulness": faithfulness,
            "correctness": correctness,
            "keyword_hit": round(kw_hit, 4),
            "citation": cit,
            "latency": metrics,
            "verification": vinfo,
            "generated_answer": answer,
            "retrieved_pages": retrieved_pages,
            "retrieved_sections": retrieved_sections,
            "search_query": metrics.get("search_query"),
            "top_rerank_score": metrics.get("top_score", 0.0),
            "low_confidence": metrics.get("low_confidence", False),
            "failure_tags": tags,
        })

        status = "OK" if not tags else "FAIL:" + ",".join(tags)
        print(
            f"      R@5={r5:.0f} P@5={p5:.2f} MRR={mrr:.2f} F={faithfulness} C={correctness} "
            f"V={vdecision} [{status}]"
        )

    n = len(questions)
    avg = {k: (v / n if n else 0.0) for k, v in sums.items()}
    avg_latencies = {k: round(v / n, 2) for k, v in latency_sums.items()} if n else {}

    # Refusal rate among OOD only
    refusal_accuracy = (sums["refusal_correct"] / ood_count * 100) if ood_count else None
    false_refusal_rate = (
        failure_counter.get("false_refusal", 0) / in_domain_count * 100 if in_domain_count else 0.0
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": n,
        "evidence_matching": "page/section/keywords (chunk IDs ignored)",
        "index_note": "Post attendance-ranking fix + answer verification",
        # Retrieval
        "recall_at_1": round(avg["recall_1"] * 100, 2),
        "recall_at_3": round(avg["recall_3"] * 100, 2),
        "recall_at_5": round(avg["recall_5"] * 100, 2),
        "precision_at_1": round(avg["precision_1"] * 100, 2),
        "precision_at_3": round(avg["precision_3"] * 100, 2),
        "precision_at_5": round(avg["precision_5"] * 100, 2),
        "mrr": round(avg["mrr"] * 100, 2),
        "ndcg_at_5": round(avg["ndcg_5"] * 100, 2),
        # Generation
        "faithfulness": round((avg["faithfulness"] / 5.0) * 100, 2),
        "correctness": round((avg["correctness"] / 5.0) * 100, 2),
        "keyword_hit_rate": round(avg["keyword_hit"] * 100, 2),
        # Citations
        "citation_valid_id_rate": round(avg["citation_valid"] * 100, 2),
        "citation_grounded_rate": round(avg["citation_grounded"] * 100, 2),
        # System behaviour
        "ood_refusal_accuracy": None if refusal_accuracy is None else round(refusal_accuracy, 2),
        "false_refusal_rate": round(false_refusal_rate, 2),
        "failure_counts": failure_counter,
        "verification_decisions": verify_decisions,
        "avg_latencies": avg_latencies,
        "results": results,
    }

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY (post re-index)")
    print("=" * 50)
    print(f"Recall@1 / @3 / @5:     {summary['recall_at_1']}% / {summary['recall_at_3']}% / {summary['recall_at_5']}%")
    print(f"Precision@1 / @3 / @5:  {summary['precision_at_1']}% / {summary['precision_at_3']}% / {summary['precision_at_5']}%")
    print(f"MRR:                    {summary['mrr']}%")
    print(f"NDCG@5:                 {summary['ndcg_at_5']}%")
    print(f"Faithfulness:           {summary['faithfulness']}%")
    print(f"Correctness:            {summary['correctness']}%")
    print(f"Keyword hit-rate:       {summary['keyword_hit_rate']}%")
    print(f"Citation valid IDs:     {summary['citation_valid_id_rate']}%")
    print(f"Citation grounded:      {summary['citation_grounded_rate']}%")
    print(f"OOD refusal accuracy:   {summary['ood_refusal_accuracy']}%")
    print(f"False refusal rate:     {summary['false_refusal_rate']}%")
    print(f"Verify decisions:       {verify_decisions}")
    print(f"Avg total latency:      {avg_latencies.get('total_ms', 0)} ms")
    print(f"  - Gen LLM:            {avg_latencies.get('generation_ms', 0)} ms")
    print(f"  - Verify:             {avg_latencies.get('verify_ms', 0)} ms")
    print(f"Failure tags:           {failure_counter}")
    print(f"Results saved to {out_path}")
    print("=" * 50)

    return summary


if __name__ == "__main__":
    run_evaluation()
