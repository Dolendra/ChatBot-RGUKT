# Manual / API smoke-test results

Date: 2026-09-05  
Model: `qwen/qwen3.8-27b`  
Pipeline: frozen RAG + verification (no code changes for this run)

## Attendance regression (recorded)

```text
Model: qwen/qwen3.8-27b
Query: minimum attendance requirement
Expected: 75%
Observed: 75%
Status: PASS
```

## Full final smoke (API mirror of UI checklist)

Results filled by `scripts/final_smoke.py` against `http://127.0.0.1:8000`.

| # | Check | Expected | Status |
| --- | --- | --- | --- |
| 1 | Minimum attendance | 75% | **PASS** |
| 2 | Consequence below 40% | admission cancel / leave / certificates | **PASS** |
| 3 | Follow-up (“below that”) | consequence of falling below prior threshold | **PASS** |
| 4 | Exact-number eligibility | 75% | **PASS** |
| 5 | Out-of-domain | abstain (`couldn't find` / `don't know from the documents`) | **PASS** |
| 6 | Citation / sources | answer has `[Source n]`; sources include attendance page | **PASS** |

### UI-only (manual)

- Click `[Source n]` → corresponding source card opens/highlights
- Refresh → conversation persists (`localStorage` + `schemaVersion`)
- Staged loading stages appear while waiting

**Overall:** ALL PASS (API). UI click / refresh / staged loading remain a short manual check at http://127.0.0.1:5173/.

### Deployment note

Deployed `GROQ_MODEL` (or provider model id) must be a model **actually available** on the account. Treat `llama` 404 / empty `gpt-oss` responses as **config failures**, not RAG regressions.

### Observed answers (API)

**1. What is the minimum attendance requirement?**

```text
The minimum attendance requirement is 75% [Source 1].
```

**2. What happens if a student's attendance falls below 40%?**

```text
If a student's attendance falls below 40%, the consequences depend on the specific program:

* **First Year of PUC:** The student will lose their admission and must leave the university with their SSC certificates only [Source 1].
* **First Year of 4-Year B.Tech (after PUC):** The student will lose their admission from the concerned B.Tech program and must leave the university with their PUC certificates [Source 1].

Note that students with attendance between 40% and the required minimum (but no
```

**3. What happens if it is below that?**

```text
If a candidate has less than 40% attendance in the first year of PUC or a 4-Year B.Tech program, they will lose their admission and must leave the university with their SSC or PUC certificates, respectively [Source 2]. Candidates with attendance between 40% and 75% may seek readmission into the same class during the subsequent year to fulfill the attendance requirements and appear for the examination, subject to payment of the requisite fee [Source 2].
```

**4. What percentage attendance is required to be eligible?**

```text
A regular program of study requires attendance of not less than 75% [Source 1].
```

**5. Who won the FIFA World Cup in 2022?**

```text
I couldn't find any relevant information in the RGUKT documents.
```

**6. citation from #1**

```text
cites=True pages=[3, 4, 7] n_sources=3
```

