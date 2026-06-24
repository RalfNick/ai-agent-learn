# Phase5 Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal observability layer to the Phase5 FastAPI Agent service so each request can be traced, measured, and inspected.

**Architecture:** Keep the production code inside `01-fastapi-backend` because observability belongs to the running API boundary. Add an in-memory `ObservabilityStore`, a FastAPI middleware for HTTP trace/latency, Agent-route instrumentation for runtime trace/tool/evidence/review/cost data, and two read endpoints for summary and trace detail. Document the learning output under `03-observability` and `docs/phase-5/`.

**Tech Stack:** Python 3.12, FastAPI middleware, Pydantic schemas, standard-library logging, unittest, SVG diagrams.

---

### Task 1: Observability API Tests

**Files:**
- Modify: `phase-5-production/01-fastapi-backend/tests/test_api.py`

- [x] **Step 1: Write failing tests**

Add tests that assert:

```python
self.assertTrue(response.headers["x-trace-id"])
self.assertEqual(summary_response.status_code, 200)
self.assertEqual(trace_response.status_code, 200)
self.assertIn("runtime.start", trace["agent"]["runtime_trace"])
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/01-fastapi-backend/tests
```

Expected: FAIL because there is no trace header and no observability endpoints.

### Task 2: Implement Observability Runtime

**Files:**
- Create: `phase-5-production/01-fastapi-backend/app/observability.py`
- Modify: `phase-5-production/01-fastapi-backend/app/schemas.py`
- Modify: `phase-5-production/01-fastapi-backend/app/main.py`

- [x] **Step 1: Add in-memory store**

Create dataclasses for `HttpObservation` and `AgentRunObservation`, plus an `ObservabilityStore` with `summary()` and `trace_detail(trace_id)`.

- [x] **Step 2: Add FastAPI middleware**

Generate or accept `X-Trace-Id`, measure request latency, add `X-Trace-Id` to the response, and record method/path/status/latency.

- [x] **Step 3: Instrument Agent answer route**

Measure Agent runtime latency and record session id, runtime trace, tool count, evidence count, review status, and deterministic estimated cost.

- [x] **Step 4: Add query endpoints**

Expose:

```text
GET /api/v1/observability/summary
GET /api/v1/observability/traces/{trace_id}
```

### Task 3: Documentation and Verification

**Files:**
- Create: `phase-5-production/03-observability/README.md`
- Create: `docs/phase-5/03-observability.md`
- Create: `docs/phase-5/diagram/observability/observability-architecture.svg`
- Modify: `phase-5-production/README.md`
- Modify: `docs/phase-5/README.md`

- [ ] **Step 1: Document runtime usage**

Show curl examples for `/api/v1/agent/answer`, `/api/v1/observability/summary`, and `/api/v1/observability/traces/{trace_id}`.

- [ ] **Step 2: Add architecture article**

Explain why Agent observability needs trace, latency, cost, evidence, review, and failure boundaries rather than only access logs.

- [ ] **Step 3: Run verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/01-fastapi-backend/tests
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/02-docker-deploy/tests
PYTHONPYCACHEPREFIX=/private/tmp/ai-agent-learn-pycache python3 -m py_compile phase-5-production/01-fastapi-backend/app/*.py
```

Expected: all commands pass.
