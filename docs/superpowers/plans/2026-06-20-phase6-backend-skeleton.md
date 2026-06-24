# Phase6 Backend Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable Phase6 capstone slice: a FastAPI backend skeleton with stable schemas, answer placeholder, observability summary, tests, and documentation.

**Architecture:** Create `phase-6-capstone/01-backend-skeleton/` as an independent FastAPI subproject. The app exposes `/health`, `/api/v1/answer`, and `/api/v1/observability/summary`; `/api/v1/answer` returns a clearly labeled placeholder response with trace/sources fields so later RAG and LangGraph modules can replace the runtime without breaking the API contract.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, Uvicorn, standard-library unittest.

## Global Constraints

- Do not implement real RAG, embeddings, vector store, or LangGraph in this slice.
- Keep the service self-contained under `phase-6-capstone/01-backend-skeleton/`.
- Every response model should already include fields needed by later sources and trace UI.
- Tests must fail before implementation and pass after implementation.

---

### Task 1: API Contract Tests

**Files:**
- Create: `phase-6-capstone/01-backend-skeleton/tests/test_api.py`

**Interfaces:**
- Consumes: `create_app()` from `app.main`.
- Produces: tests defining `/health`, `/api/v1/answer`, validation errors, and `/api/v1/observability/summary`.

- [x] **Step 1: Write failing tests**

```python
class Phase6BackendApiTests(unittest.TestCase):
    def test_health_returns_capstone_metadata(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["phase"], "phase-6")

    def test_answer_returns_placeholder_contract(self) -> None:
        response = self.client.post("/api/v1/answer", json={"question": "Phase6 要做什么？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "placeholder")
        self.assertIn("trace", response.json())
        self.assertIn("sources", response.json())

    def test_observability_summary_counts_answer_requests(self) -> None:
        self.client.post("/api/v1/answer", json={"question": "Phase6 要做什么？"})
        response = self.client.get("/api/v1/observability/summary")
        self.assertEqual(response.json()["total_answer_requests"], 1)
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/01-backend-skeleton/tests
```

Expected: FAIL because `app.main` does not exist yet.

### Task 2: FastAPI Skeleton

**Files:**
- Create: `phase-6-capstone/01-backend-skeleton/app/__init__.py`
- Create: `phase-6-capstone/01-backend-skeleton/app/config.py`
- Create: `phase-6-capstone/01-backend-skeleton/app/schemas.py`
- Create: `phase-6-capstone/01-backend-skeleton/app/runtime.py`
- Create: `phase-6-capstone/01-backend-skeleton/app/observability.py`
- Create: `phase-6-capstone/01-backend-skeleton/app/main.py`
- Create: `phase-6-capstone/01-backend-skeleton/requirements.txt`

**Interfaces:**
- Produces: `create_app(settings: Settings | None = None) -> FastAPI`.
- Produces: `AnswerRuntime.answer(question: str, session_id: str) -> AnswerResponse`.
- Produces: `ObservabilityStore.summary() -> dict`.

- [x] **Step 1: Implement Pydantic schemas**

Create health, answer request/response, source, trace step, and observability summary models.

- [x] **Step 2: Implement placeholder runtime**

Return an answer that states the backend skeleton is ready and real retrieval will arrive in `02-knowledge-ingestion` and `03-agentic-qa-runtime`.

- [x] **Step 3: Implement observability store**

Track answer request count, last trace id, and recent questions in memory.

- [x] **Step 4: Implement FastAPI app**

Expose `/health`, `/api/v1/answer`, and `/api/v1/observability/summary`.

### Task 3: Documentation and Verification

**Files:**
- Create: `phase-6-capstone/01-backend-skeleton/README.md`
- Modify: `phase-6-capstone/README.md`
- Create: `docs/phase-6/01-backend-skeleton.md`
- Modify: `docs/phase-6/README.md`

**Interfaces:**
- Consumes: successful test output and smoke response.
- Produces: first Phase6 implementation article and updated learning index.

- [x] **Step 1: Document run commands**

Show install, test, uvicorn, `/health`, `/api/v1/answer`, and `/api/v1/observability/summary`.

- [x] **Step 2: Write Phase6 article**

Explain why the capstone starts with service boundary before RAG and LangGraph.

- [ ] **Step 3: Run verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/01-backend-skeleton/tests
PYTHONPYCACHEPREFIX=/private/tmp/ai-agent-learn-pycache python3 -m py_compile phase-6-capstone/01-backend-skeleton/app/*.py
```

Expected: all commands pass.
