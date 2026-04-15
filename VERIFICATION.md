# Engram — DoclingClient Verification Record

**Status:** Pending manual verification (E7 integration gate).

This file records the verified Docling-serve API behavior that `engram/clients/docling.py`
is written against. Update this file whenever:
- E7 integration tests pass for the first time (fill in the table below)
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

Fill in after manual verification. Check each box once confirmed.

| Assumption | Expected value | Verified value | Confirmed |
|---|---|---|---|
| Health endpoint | `GET /health` → 200 | | [ ] |
| Convert endpoint | `POST /v1/convert/file/async` | | [ ] |
| Chunk hybrid endpoint | `POST /v1/chunk/hybrid/file/async` | | [ ] |
| Chunk hierarchical endpoint | `POST /v1/chunk/hierarchical/file/async` | | [ ] |
| Status poll endpoint | `GET /v1/status/poll/{task_id}?wait=30` | | [ ] |
| Result endpoint | `GET /v1/result/{task_id}` | | [ ] |
| Clear results endpoint | `GET /v1/clear/results?older_then=N` | | [ ] |
| Submit response field | `task_id` at top level | | [ ] |
| Status field name | `task_status` | | [ ] |
| Status success value | `"success"` | | [ ] |
| Status failure value | `"failure"` | | [ ] |
| Status pending value | `"pending"` or `"started"` | | [ ] |
| Error field on failure | `error_message` | | [ ] |
| Convert result path | `result.document.md_content` | | [ ] |
| Chunk list path | `result.chunks` | | [ ] |
| Chunk text field | `chunk["text"]` | | [ ] |
| Corrupted file → failure | `task_status: "failure"` with error_message | | [ ] |

---

## Docling-serve version pinned

| Field | Value |
|---|---|
| Image | `quay.io/docling-project/docling-serve:latest` |
| Version verified | (fill in from `GET /version`) |
| Date verified | |
| Verified by | |

---

## Known deviations from plan assumptions

None recorded yet. Document here if any verified value differs from the "Expected value" column above,
along with the corresponding fix applied in `engram/clients/docling.py`.
