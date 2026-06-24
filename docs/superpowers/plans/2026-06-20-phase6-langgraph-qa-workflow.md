# Phase6 LangGraph QA Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Phase6 deterministic Agentic QA runtime into a real LangGraph workflow with explicit retrieve, context grade, answer generation, evidence review, repair, and abstain routes.

**Architecture:** Keep `AgenticQARuntime` as the public API. Internally it delegates to a LangGraph `StateGraph` built in `agentic_qa/workflow.py`. Nodes stay deterministic and LLM-free, but the routing shape now matches the later model-powered Agentic RAG runtime.

**Tech Stack:** Python 3.11+, LangGraph, Phase6 local knowledge index, `unittest`.

## Global Constraints

- Keep existing API and CLI contracts stable.
- Do not call LLM APIs in this slice.
- Use real `langgraph.graph.StateGraph`; do not simulate graph routing.
- Keep repair and abstain paths deterministic and testable.
- Preserve evidence-only answer generation as the safe fallback.

---

### Task 1: LangGraph Workflow Tests

**Files:**
- Modify: `phase-6-capstone/03-agentic-qa-runtime/tests/test_agentic_qa_runtime.py`

**Interfaces:**
- `AgenticQARuntime(..., unsafe_answer_builder: Callable | None = None, max_repairs: int = 1)`
- `response.trace` should include `review.failed` and `answer.repair` when the answer builder returns unsupported text.
- Weak context should still route to `abstain`.

- [x] **Step 1: Write failing graph route tests**

Add tests for:
- repair route removes unsupported answer lines.
- graph trace includes `review.failed` then `answer.repair`.
- weak context path still ends at `abstain`.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/03-agentic-qa-runtime/tests
```

Expected: fail because `unsafe_answer_builder` is not accepted or repair trace does not exist.

### Task 2: LangGraph Workflow Implementation

**Files:**
- Create: `phase-6-capstone/03-agentic-qa-runtime/agentic_qa/workflow.py`
- Modify: `phase-6-capstone/03-agentic-qa-runtime/agentic_qa/runtime.py`

**Interfaces:**
- `build_qa_workflow(resources: WorkflowResources)`
- `WorkflowResources` contains index, threshold, top_k, max_repairs, answer_builder.
- Graph route:

```text
retrieve -> context_grade
  weak   -> abstain
  enough -> answer_generate -> evidence_review
      supported -> END
      failed    -> repair -> evidence_review
      failed after repair -> abstain
```

- [x] **Step 1: Implement graph**

Use `StateGraph` and conditional edges.

- [x] **Step 2: Verify GREEN**

Run runtime tests.

### Task 3: Docs And Review

**Files:**
- Modify: `phase-6-capstone/03-agentic-qa-runtime/README.md`
- Modify: `docs/phase-6/03-agentic-qa-runtime.md`

- [x] **Step 1: Document workflow graph**

Explain that this is real LangGraph routing but still deterministic answer generation.

- [x] **Step 2: Round review**

Review:
- StateGraph route visibility.
- repair path is testable.
- limitations before adding LLM generation.

### Task 4: Verification

- [x] **Step 1: Runtime tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/03-agentic-qa-runtime/tests
```

- [x] **Step 2: Backend tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/01-backend-skeleton/tests
```

- [x] **Step 3: Compile**

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-agent-learn-pycache python3 -m py_compile phase-6-capstone/03-agentic-qa-runtime/agentic_qa/*.py phase-6-capstone/03-agentic-qa-runtime/*.py
```

- [x] **Step 4: CLI smoke**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase-6-capstone/03-agentic-qa-runtime/run_agentic_qa.py --source docs/phase-6 --question "Phase6 为什么需要 trace？"
```
