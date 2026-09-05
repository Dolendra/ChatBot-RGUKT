"""
Final product smoke: attendance, consequence, follow-up, OOD, citations.
Requires API up. Does not modify RAG code.

  python scripts/final_smoke.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8000")
ABSTAIN = "I don't know from the documents."


def chat(message: str, history: list | None = None, timeout: float = 180.0) -> dict:
    payload = {"message": message, "top_k": 3, "history": history or []}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {raw[:300]}") from e


def pages(sources: list) -> list:
    out = []
    for s in sources or []:
        p = s.get("page")
        if p is not None:
            out.append(p)
    return out


def has_cite(answer: str) -> bool:
    return bool(re.search(r"\[Source\s+\d+\]", answer or "", re.I))


def main() -> int:
    rows: list[tuple[str, bool, str]] = []

    # 1 Minimum attendance
    q1 = "What is the minimum attendance requirement?"
    r1 = chat(q1)
    a1 = r1.get("answer") or ""
    ok1 = "75" in a1 and "%" in a1
    rows.append(("1 min attendance -> 75%", ok1, a1[:160].replace("\n", " ")))

    # 6 Citation (from r1)
    src_pages = pages(r1.get("sources") or [])
    ok6 = has_cite(a1) and len(r1.get("sources") or []) > 0
    cite_note = f"cites={has_cite(a1)} pages={src_pages} n_sources={len(r1.get('sources') or [])}"
    rows.append(("6 citation markers + sources", ok6, cite_note))

    # 2 Consequence below 40%
    q2 = "What happens if a student's attendance falls below 40%?"
    r2 = chat(q2)
    a2 = (r2.get("answer") or "").lower()
    ok2 = any(
        k in a2
        for k in ("cancel", "admission", "leave", "certificate", "ssc", "detained")
    )
    rows.append(("2 below 40% consequence", ok2, (r2.get("answer") or "")[:160].replace("\n", " ")))

    # 3 Follow-up after attendance (conversational rewrite)
    hist = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": a1},
    ]
    q3 = "What happens if it is below that?"
    r3 = chat(q3, history=hist)
    a3 = (r3.get("answer") or "").lower()
    ok3 = any(
        k in a3
        for k in ("detained", "examination", "readmission", "40%", "attendance", "barred")
    ) and ABSTAIN.lower() not in a3
    rows.append(("3 follow-up below that", ok3, (r3.get("answer") or "")[:160].replace("\n", " ")))

    # 4 Exact-number eligibility
    q4 = "What percentage attendance is required to be eligible?"
    r4 = chat(q4)
    a4 = r4.get("answer") or ""
    ok4 = "75" in a4
    rows.append(("4 exact-number eligible -> 75%", ok4, a4[:160].replace("\n", " ")))

    # 5 OOD
    q5 = "Who won the FIFA World Cup in 2022?"
    r5 = chat(q5)
    a5 = (r5.get("answer") or "").strip()
    a5_l = a5.lower()
    ok5 = (
        a5_l.startswith("i don't know from the documents")
        or "couldn't find" in a5_l
        or "don't know" in a5_l
    )
    rows.append(("5 OOD abstain", ok5, a5[:120]))

    print("=" * 60)
    print("FINAL SMOKE RESULTS")
    print("=" * 60)
    failed = 0
    for name, ok, info in rows:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        safe = (info or "").encode("ascii", "replace").decode("ascii")
        print(f"[{mark}] {name}")
        print(f"       {safe}")
    print("=" * 60)
    print(f"{'ALL PASS' if failed == 0 else f'{failed} FAILED'} ({len(rows)} checks)")

    # Update SMOKE_TEST.md status table if present
    md_path = os.path.join(ROOT, "data", "SMOKE_TEST.md")
    if os.path.exists(md_path):
        status_map = {
            "1": "PASS" if rows[0][1] else "FAIL",
            "2": "PASS" if rows[2][1] else "FAIL",
            "3": "PASS" if rows[3][1] else "FAIL",
            "4": "PASS" if rows[4][1] else "FAIL",
            "5": "PASS" if rows[5][1] else "FAIL",
            "6": "PASS" if rows[1][1] else "FAIL",
        }
        text = open(md_path, encoding="utf-8").read()
        for n, st in status_map.items():
            text = re.sub(
                rf"(\| {n} \|[^\|]+\|[^\|]+\| )_pending_",
                rf"\1**{st}**",
                text,
            )
        # Append raw answers block
        block = "\n### Observed answers (API)\n\n"
        labels = [
            ("1", q1, a1),
            ("2", q2, r2.get("answer") or ""),
            ("3", q3, r3.get("answer") or ""),
            ("4", q4, a4),
            ("5", q5, a5),
            ("6", "citation from #1", cite_note),
        ]
        for n, q, a in labels:
            block += f"**{n}. {q}**\n\n```text\n{(a or '')[:500]}\n```\n\n"
        if "### Observed answers (API)" in text:
            text = text.split("### Observed answers (API)")[0].rstrip() + "\n" + block
        else:
            text = text.rstrip() + "\n" + block
        open(md_path, "w", encoding="utf-8").write(text)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
