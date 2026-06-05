# Phase5 FastAPI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the Phase4 integrated Agent runtime behind a minimal FastAPI service with typed request/response models and API tests.

**Architecture:** Keep Phase5 first step under `phase-5-production/01-fastapi-backend/`. The service imports Phase4's deterministic `IntegratedAgentRuntime` through a thin adapter, exposes health and answer endpoints, and stores demo memory under a local ignored `.memory/` directory.

**Tech Stack:** Python, FastAPI, Pydantic v2, Uvicorn, unittest, FastAPI TestClient.

---

### Task 1: Write API Contract Tests

**Files:**
- Create: `phase-5-production/01-fastapi-backend/tests/test_api.py`

- [ ] Test `GET /health` returns `ok`, service name, and phase.
- [ ] Test `POST /api/v1/agent/answer` returns answer, evidence, trace, review, memory context, and tool results.
- [ ] Test an empty question returns HTTP 422.

### Task 2: Implement FastAPI App

**Files:**
- Create: `phase-5-production/01-fastapi-backend/app/config.py`
- Create: `phase-5-production/01-fastapi-backend/app/schemas.py`
- Create: `phase-5-production/01-fastapi-backend/app/runtime_adapter.py`
- Create: `phase-5-production/01-fastapi-backend/app/main.py`
- Create: `phase-5-production/01-fastapi-backend/app/__init__.py`

- [ ] Define settings, request/response schemas, runtime adapter, and app factory.
- [ ] Keep all endpoints deterministic and local; no external model call in Phase5 step one.
- [ ] Convert Phase4 runtime dataclasses into JSON-friendly Pydantic responses.

### Task 3: Add Run Docs

**Files:**
- Create: `phase-5-production/01-fastapi-backend/requirements.txt`
- Create: `phase-5-production/01-fastapi-backend/README.md`
- Create: `phase-5-production/01-fastapi-backend/.gitignore`

- [ ] Document install, test, run, curl examples, and current boundaries.

### Task 4: Write Phase5 Opening Article

**Files:**
- Create: `docs/phase-5/01-fastapi-agent-service.md`
- Create: `docs/phase-5/README.md`

- [ ] Explain why Phase5 starts by wrapping the runtime as an API, not by jumping straight to Docker.
- [ ] Include architecture, request/response schema, test output, and production gaps.

### Task 5: Verify

- [ ] Run Phase5 backend tests.
- [ ] Run Phase4 runtime tests to ensure the adapter did not break upstream assumptions.
- [ ] Run py_compile for Phase5 backend and Phase4 integration modules.
- [ ] Validate Phase5 article H1 count and local image paths.
