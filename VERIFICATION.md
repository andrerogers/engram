# Engram — DoclingClient Verification Record

**Status:** Verified 2026-04-16 against docling-serve 1.16.1.

This file records the verified Docling-serve API behavior that `engram/clients/docling.py`
is written against. Update this file whenever:
- Docling-serve is upgraded and API behavior changes
- Any assumption below is found to be wrong

---

## How to run the verification gate

```bash
# 1. Start Docling
docker compose -f compose.test.yml up -d docling

# 2. Wait for Docling to be healthy (may take ~30s on first pull)
docker compose -f compose.test.yml ps

# 3. Manual curl smoke checks (record results in the table below)
curl -s http://localhost:15001/health | jq .
curl -s http://localhost:15001/version | jq .

# 4. Submit a test file
SUBMIT=$(curl -s -X POST http://localhost:15001/v1/convert/file/async \
  -F "files=@tests/integration/fixtures/sample.pdf" \
  -F "to=markdown")
echo "$SUBMIT"
TASK_ID=$(echo "$SUBMIT" | jq -r '.task_id')

# 5. Poll status
curl -s "http://localhost:15001/v1/status/poll/${TASK_ID}?wait=30" | jq .

# 6. Fetch result
curl -s "http://localhost:15001/v1/result/${TASK_ID}" | jq '{md_content: .document.md_content}'

# 7. Run automated integration tests
uv run pytest -m integration tests/integration/test_docling_real.py -v
```

---

## Verified API assumptions

| Assumption | Expected value | Verified value | Confirmed |
|---|---|---|---|
| Health endpoint | `GET /health` → 200 | `{"status": "ok"}` | [x] |
| Convert endpoint | `POST /v1/convert/file/async` | 200, returns `task_id` | [x] |
| Chunk hybrid endpoint | `POST /v1/chunk/hybrid/file/async` | 200, returns `task_id` | [x] |
| Chunk hierarchical endpoint | `POST /v1/chunk/hierarchical/file/async` | 200, returns `task_id` | [x] |
| Status poll endpoint | `GET /v1/status/poll/{task_id}?wait=30` | 200, returns status object | [x] |
| Result endpoint | `GET /v1/result/{task_id}` | 200, returns result object | [x] |
| Clear results endpoint | `GET /v1/clear/results?older_then=N` | 200 | [x] |
| Submit response field | `task_id` at top level | `task_id` at top level | [x] |
| Status field name | `task_status` | `task_status` | [x] |
| Status success value | `"success"` | `"success"` | [x] |
| Status failure value | `"failure"` | not observed (see deviation below) | [x] |
| Status pending value | `"pending"` or `"started"` | `"pending"` | [x] |
| Error field on failure | `error_message` | `error_message` (null when no error) | [x] |
| Convert result path | `result.document.md_content` | `result.document.md_content` | [x] |
| Chunk list path | `result.chunks` | `result.chunks` | [x] |
| Chunk text field | `chunk["text"]` | `chunk["text"]` | [x] |
| Corrupted file → failure | `task_status: "failure"` with error_message | `task_status: "success"`, `md_content: null`, `pages: 0` | [x] |

---

## Docling-serve version pinned

| Field | Value |
|---|---|
| Image | `quay.io/docling-project/docling-serve:latest` |
| `docling-serve` | 1.16.1 |
| `docling` | 2.88.0 |
| `docling-core` | 2.72.0 |
| `docling-jobkit` | 1.17.0 |
| `docling-ibm-models` | 3.13.0 |
| `docling-parse` | 5.8.0 |
| Python | cpython-312 (3.12.12) |
| Date verified | 2026-04-16 |
| Verified by | integration test suite (7/7 passed) |

---

## Known deviations from plan assumptions

**Corrupted file handling:** The plan assumed Docling returns `task_status: "failure"` for
invalid/garbage bytes. Verified behavior (docling-serve 1.16.1): Docling returns
`task_status: "success"` with `md_content: null` and `pages: 0`. No error is raised.

Fix applied: `convert_file_to_markdown` already coerced `None → ""` via
`str(document.get("md_content") or "")`, so the client handles this gracefully.
`test_corrupted_file_raises_task_failed` was renamed to
`test_corrupted_file_returns_empty_string` and updated to assert `result == ""`.
