# Phase6 Agentic QA Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the Phase6 backend answer contract to a reusable Agentic QA runtime that retrieves local knowledge chunks, grades context quality, answers from evidence, and abstains when evidence is weak.

**Architecture:** Add a standalone `03-agentic-qa-runtime` sub-project that imports the deterministic `02-knowledge-ingestion` retrieval layer. Keep the first runtime LLM-free so API behavior, sources, trace, review status, and abstain routing can be tested before introducing external model calls or LangGraph.

**Tech Stack:** Python 3.11+, stdlib dataclasses/pathlib/sys, Phase6 local knowledge index, FastAPI TestClient integration tests, `unittest`.

## Global Constraints

- Use TDD: write failing tests before implementation.
- Do not call LLM APIs in this slice.
- Keep the runtime reusable outside FastAPI.
- Preserve the existing `/api/v1/answer` response schema.
- Keep `01-backend-skeleton` runnable in placeholder mode by default.
- Add an explicit injection point so tests and later slices can use `AgenticQARuntime`.

---

### Task 1: Runtime Tests

**Files:**
- Create: `phase-6-capstone/03-agentic-qa-runtime/tests/test_agentic_qa_runtime.py`

**Interfaces:**
- `AgenticQARuntime(index: LocalKnowledgeIndex, min_context_score: float = 0.25, top_k: int = 3)`
- `AgenticQARuntime.answer(question: str, session_id: str = "default") -> QAResponse`
- `build_runtime_from_sources(paths: Sequence[Path | str]) -> AgenticQARuntime`

- [x] **Step 1: Write failing tests**

Tests assert:
- A grounded question returns `mode="agentic_rag"` with sources and retrieve/context/answer trace steps.
- An unrelated question returns `review_status="abstained"` and no hallucinated answer.
- `build_runtime_from_sources` can load a temporary docs folder and answer from it.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/03-agentic-qa-runtime/tests
```

Expected: import failure because `agentic_qa` does not exist.

### Task 2: Runtime Implementation

**Files:**
- Create: `phase-6-capstone/03-agentic-qa-runtime/agentic_qa/__init__.py`
- Create: `phase-6-capstone/03-agentic-qa-runtime/agentic_qa/models.py`
- Create: `phase-6-capstone/03-agentic-qa-runtime/agentic_qa/runtime.py`
- Create: `phase-6-capstone/03-agentic-qa-runtime/run_agentic_qa.py`
- Create: `phase-6-capstone/03-agentic-qa-runtime/requirements.txt`
- Create: `phase-6-capstone/03-agentic-qa-runtime/.gitignore`

**Interfaces:**
- `QAResponse` contains `question`, `session_id`, `answer`, `mode`, `sources`, `trace`, `review_status`, `context_score`.
- `QASource` contains `source_id`, `title`, `path`, `score`, `snippet`.
- `QATraceStep` contains `step`, `detail`, `latency_ms`.

- [x] **Step 1: Implement minimal runtime**

Route:

```text
request.received -> retrieve -> context_grade
  high enough -> answer.generate -> review.evidence_supported
  weak        -> abstain
```

Answer generation should be evidence-only:

```text
根据当前知识库资料，可以确认：
1. ...
```

- [x] **Step 2: Verify GREEN**

Run runtime tests and ensure they pass.

### Task 3: Backend Integration

**Files:**
- Modify: `phase-6-capstone/01-backend-skeleton/app/main.py`
- Modify: `phase-6-capstone/01-backend-skeleton/app/runtime.py`
- Modify: `phase-6-capstone/01-backend-skeleton/tests/test_api.py`

**Interfaces:**
- `create_app(settings: Settings | None = None, runtime: AnswerRuntimeProtocol | None = None) -> FastAPI`
- Runtime object must expose `answer(question: str, session_id: str) -> AnswerResponse`

- [x] **Step 1: Write failing API integration test**

Test passes an adapter runtime into `create_app(runtime=...)` and asserts `/api/v1/answer` returns `mode="agentic_rag"` with real sources.

- [x] **Step 2: Implement runtime injection**

Keep default behavior as placeholder. Allow tests and later app assembly to inject Agentic runtime.

- [x] **Step 3: Verify backend tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/01-backend-skeleton/tests
```

### Task 4: Documentation And Round Review

**Files:**
- Create: `phase-6-capstone/03-agentic-qa-runtime/README.md`
- Create: `docs/phase-6/03-agentic-qa-runtime.md`
- Modify: `docs/phase-6/README.md`
- Modify: `phase-6-capstone/README.md`

- [x] **Step 1: Document design**

Explain that this slice proves the QA control flow and API integration, not final LLM answer quality.

- [x] **Step 2: Round review**

Review for:
- contract compatibility with `01-backend-skeleton`.
- reuse of `02-knowledge-ingestion`.
- clear next step into Web UI or LLM/LangGraph upgrade.

### Task 5: Verification

- [x] **Step 1: Runtime tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/03-agentic-qa-runtime/tests
```

- [x] **Step 2: Backend tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/01-backend-skeleton/tests
```

- [x] **Step 3: Knowledge ingestion regression**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/02-knowledge-ingestion/tests
```

- [x] **Step 4: Compile**

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-agent-learn-pycache python3 -m py_compile phase-6-capstone/03-agentic-qa-runtime/agentic_qa/*.py phase-6-capstone/03-agentic-qa-runtime/*.py phase-6-capstone/01-backend-skeleton/app/*.py
```

- [x] **Step 5: CLI smoke**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase-6-capstone/03-agentic-qa-runtime/run_agentic_qa.py --source docs/phase-6 --question "Phase6 为什么需要 trace？"
```
