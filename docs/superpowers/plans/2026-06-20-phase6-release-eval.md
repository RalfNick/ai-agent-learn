# Phase6 Release Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Phase6 release/eval slice that can run deterministic golden-set checks and provide a reproducible backend/frontend compose entry for the capstone demo.

**Architecture:** Create `phase-6-capstone/05-release-eval`. The eval runner uses the existing LangGraph QA runtime directly for stable smoke checks. The integrated API entry wires `01-backend-skeleton` to `03-agentic-qa-runtime` with an adapter. Docker Compose defines backend and web services without changing earlier sub-project independence.

**Tech Stack:** Python 3.11+, unittest, FastAPI/Uvicorn, Docker Compose, Next.js app from `04-web-ui`.

## Global Constraints

- Keep eval deterministic and local.
- Do not call external LLM APIs.
- Keep backend and web independently runnable.
- Compose should describe the release path even if full image build is left for the final production hardening pass.

---

### Task 1: Release Eval Tests

**Files:**
- Create: `phase-6-capstone/05-release-eval/tests/test_release_eval.py`

- [x] **Step 1: Write failing tests**

Tests assert:
- `evaluate_cases` returns pass/fail records with pass rate.
- A case requiring `trace` passes against temp docs.
- Missing expected evidence fails clearly.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/05-release-eval/tests
```

Expected: import failure because `release_eval` does not exist.

### Task 2: Eval Runner And API Entry

**Files:**
- Create: `phase-6-capstone/05-release-eval/release_eval/__init__.py`
- Create: `phase-6-capstone/05-release-eval/release_eval/evaluator.py`
- Create: `phase-6-capstone/05-release-eval/api_server.py`
- Create: `phase-6-capstone/05-release-eval/run_eval.py`
- Create: `phase-6-capstone/05-release-eval/eval_cases.json`
- Create: `phase-6-capstone/05-release-eval/requirements.txt`

- [x] **Step 1: Implement eval runner**

Use `build_runtime_from_sources` and check:
- `review_status`.
- expected terms in answer.
- expected source title when provided.

- [x] **Step 2: Implement integrated API entry**

Build FastAPI app with `create_app(runtime=AgenticRuntimeAdapter(...))`.

### Task 3: Compose And Docs

**Files:**
- Create: `phase-6-capstone/05-release-eval/docker-compose.yml`
- Create: `phase-6-capstone/05-release-eval/README.md`
- Create: `docs/phase-6/05-release-eval.md`
- Modify: `docs/phase-6/README.md`
- Modify: `phase-6-capstone/README.md`

- [x] **Step 1: Add compose**

Services:
- backend: Uvicorn integrated API on 8010.
- web: Next.js dev server on 3020.

- [x] **Step 2: Add docs and final review**

Explain current release scope and remaining production gaps.

### Task 4: Verification

- [x] **Step 1: Eval tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/05-release-eval/tests
```

- [x] **Step 2: Eval CLI**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase-6-capstone/05-release-eval/run_eval.py --source docs/phase-6 --cases phase-6-capstone/05-release-eval/eval_cases.json
```

- [x] **Step 3: Compose config**

```bash
cd phase-6-capstone/05-release-eval && docker compose config
```
