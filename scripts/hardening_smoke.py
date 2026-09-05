"""
Hardening smoke tests: validation, health, and user-facing error shapes.
Run with API up:  python scripts/hardening_smoke.py
Or unit-only:      python scripts/hardening_smoke.py --offline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

API = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8000")


def _post(path: str, payload: dict, timeout: float = 30.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            detail = {"detail": raw}
        return e.code, detail


def _get(path: str, timeout: float = 10.0):
    req = urllib.request.Request(f"{API}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            detail = {"detail": raw}
        return e.code, detail
    except urllib.error.URLError as e:
        return None, {"detail": str(e.reason)}


def offline_checks() -> list[tuple[str, bool, str]]:
    results = []

    # Index / embedding presence
    for path in ("embeddings.pkl", "faiss_index.index"):
        ok = os.path.exists(os.path.join(ROOT, path))
        results.append((f"artifact:{path}", ok, "present" if ok else "MISSING"))

    # Env hygiene
    gitignore = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    results.append((".gitignore has .env", ".env" in gitignore, "ok" if ".env" in gitignore else "missing"))

    # Frontend must not embed secrets
    fe_src = os.path.join(ROOT, "frontend", "src")
    secret_hits = []
    for dirpath, _, files in os.walk(fe_src):
        for f in files:
            if not f.endswith((".js", ".jsx", ".ts", ".tsx")):
                continue
            text = open(os.path.join(dirpath, f), encoding="utf-8", errors="ignore").read()
            for needle in ("GROQ_API_KEY", "gsk_", "sk-"):
                if needle in text:
                    secret_hits.append(f"{f}:{needle}")
    results.append(("frontend has no API secrets", not secret_hits, ", ".join(secret_hits) or "clean"))

    # ChatStore schema versioning
    store = open(os.path.join(ROOT, "frontend", "src", "chatStore.js"), encoding="utf-8").read()
    results.append(("chatStore schemaVersion", "SCHEMA_VERSION" in store, "present" if "SCHEMA_VERSION" in store else "missing"))

    # Validation constraints exist on API models
    api = open(os.path.join(ROOT, "api_server.py"), encoding="utf-8").read()
    results.append(("chat message max_length", "max_length=8000" in api, "ok"))
    results.append(("top_k bounds", "ge=1, le=12" in api or "le=12" in api, "ok"))
    results.append(("user-facing error constants", "_USER_UNAVAILABLE" in api, "ok"))

    return results


def online_checks() -> list[tuple[str, bool, str]]:
    results = []
    status, health = _get("/api/health")
    results.append(("GET /api/health", status == 200, str(health)[:120]))

    status, cfg = _get("/api/config")
    results.append(("GET /api/config", status == 200 and "app_title" in cfg, str(cfg)[:120]))

    # Empty / whitespace message → validation error (not 500)
    status, body = _post("/api/chat", {"message": "   ", "top_k": 3})
    detail = body.get("detail")
    detail_s = detail if isinstance(detail, str) else json.dumps(detail)[:200]
    results.append(
        (
            "whitespace message rejected safely",
            status in (422, 400) and "Traceback" not in detail_s,
            f"{status}: {detail_s}",
        )
    )

    # Invalid top_k
    status, body = _post("/api/chat", {"message": "hello", "top_k": 99})
    detail = body.get("detail")
    detail_s = detail if isinstance(detail, str) else json.dumps(detail)[:200]
    results.append(
        (
            "invalid top_k rejected",
            status == 422,
            f"{status}: {detail_s}",
        )
    )

    # Malformed history role
    status, body = _post(
        "/api/chat",
        {"message": "hello", "top_k": 3, "history": [{"role": "system", "content": "x"}]},
    )
    results.append(("malformed history rejected", status == 422, f"status={status}"))

    # Very long query truncated by validation
    status, body = _post("/api/chat", {"message": "x" * 9000, "top_k": 3})
    results.append(("oversized message rejected", status == 422, f"status={status}"))

    # Happy path smoke (may take a while)
    status, body = _post(
        "/api/chat",
        {"message": "What is the minimum attendance requirement?", "top_k": 3, "history": []},
        timeout=120.0,
    )
    ok = status == 200 and isinstance(body.get("answer"), str) and "metrics" in body
    # Must not leak stack traces
    leaked = "Traceback" in json.dumps(body)
    results.append(("chat happy path", ok and not leaked, f"status={status}, leaked={leaked}"))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    all_rows = offline_checks()
    if not args.offline:
        all_rows.extend(online_checks())

    print("=" * 60)
    print("HARDENING SMOKE RESULTS")
    print("=" * 60)
    failed = 0
    for name, ok, info in all_rows:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{mark}] {name} — {info}")
    print("=" * 60)
    print(f"{'ALL PASS' if failed == 0 else f'{failed} FAILED'} ({len(all_rows)} checks)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
