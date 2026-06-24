# Phase5 Testing Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal API regression and evaluation harness for the Phase5 FastAPI Agent service.

**Architecture:** Keep the evaluation entrypoint inside `01-fastapi-backend` so it reuses the same RuntimeAdapter and observability path as real traffic. The evaluation runner owns built-in regression cases, replays them through the Agent runtime, checks review status, trace steps, tool names, and evidence thresholds, then returns pass rate and per-case diagnostics. Phase5 `04-testing-eval` documents how to run and interpret the harness.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, unittest, SVG diagrams.

## Global Constraints

- No external evaluation SaaS or RAGAS dependency in this step.
- Evaluation must call the same runtime adapter used by `/api/v1/agent/answer`.
- Evaluation results must be inspectable through observability trace detail.
- Tests must fail before implementation and pass after implementation.

---

### Task 1: API Contract Tests

**Files:**
- Modify: `phase-5-production/01-fastapi-backend/tests/test_api.py`

**Interfaces:**
- Consumes: existing `create_app()` from `app.main`.
- Produces: tests for `GET /api/v1/evaluations/cases` and `POST /api/v1/evaluations/run`.

- [x] **Step 1: Write failing tests**

```python
response = self.client.get("/api/v1/evaluations/cases")
self.assertEqual(response.status_code, 200)

response = self.client.post(
    "/api/v1/evaluations/run",
    json={"case_ids": ["phase4-memory-evidence"], "session_prefix": "eval-test"},
)
self.assertEqual(response.status_code, 200)
self.assertEqual(response.json()["pass_rate"], 1.0)
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/01-fastapi-backend/tests
```

Expected: FAIL with 404 for evaluation endpoints.

### Task 2: Evaluation Runner

**Files:**
- Create: `phase-5-production/01-fastapi-backend/app/evaluation.py`
- Modify: `phase-5-production/01-fastapi-backend/app/schemas.py`
- Modify: `phase-5-production/01-fastapi-backend/app/main.py`

**Interfaces:**
- Consumes: `RuntimeAdapter.answer(question: str, session_id: str) -> AnswerResponse`.
- Produces: `EvaluationRunner.list_cases() -> list[dict]` and `EvaluationRunner.run(case_ids: list[str] | None, session_prefix: str) -> dict`.

- [x] **Step 1: Implement built-in evaluation cases**

Use cases covering memory evidence, code architecture, and observability readiness.

- [x] **Step 2: Implement deterministic judge**

Check:

```text
review_status
minimum_evidence_count
required_tool_names
required_trace_steps
```

- [x] **Step 3: Add FastAPI endpoints**

Expose:

```text
GET /api/v1/evaluations/cases
POST /api/v1/evaluations/run
```

- [x] **Step 4: Record eval runs in observability**

Each eval case writes an `AgentRunObservation` with trace id `eval-{case_id}`.

### Task 3: Documentation and Verification

**Files:**
- Create: `phase-5-production/04-testing-eval/README.md`
- Create: `docs/phase-5/04-testing-eval.md`
- Create: `docs/phase-5/diagram/testing-eval/testing-eval-architecture.svg`
- Modify: `phase-5-production/README.md`
- Modify: `docs/phase-5/README.md`

**Interfaces:**
- Consumes: real smoke output from `POST /api/v1/evaluations/run`.
- Produces: Phase5 closing article for API regression and evaluation.

- [ ] **Step 1: Document commands**

Show how to list cases, run all cases, run selected cases, and inspect eval trace detail.

- [ ] **Step 2: Include real smoke numbers**

Use:

```text
total_cases=3, passed_cases=3, failed_cases=0, pass_rate=1.0
```

- [ ] **Step 3: Run verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/01-fastapi-backend/tests
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/02-docker-deploy/tests
PYTHONPYCACHEPREFIX=/private/tmp/ai-agent-learn-pycache python3 -m py_compile phase-5-production/01-fastapi-backend/app/*.py
```
