# Phase6 Capstone Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start Phase6 by creating the capstone architecture, learning route, and first implementation boundary.

**Architecture:** Phase6 combines the prior phases into an enterprise knowledge-base Agent system. The first output is intentionally design-first: repository README, phase article index, architecture diagram, and system design article. Implementation starts next with `01-backend-skeleton`.

**Tech Stack:** FastAPI, LangGraph, hybrid RAG, Chroma/Milvus, Next.js, Docker Compose, observability/evaluation from Phase5.

## Global Constraints

- Do not implement all capstone modules in one step.
- Keep Phase6 scoped to a single-tenant learning system.
- Use existing Phase2-5 outputs as building blocks.
- Every later implementation slice must be independently runnable and testable.

---

### Task 1: Phase6 Design Scaffold

**Files:**
- Create: `phase-6-capstone/README.md`
- Create: `docs/phase-6/README.md`
- Create: `docs/phase-6/00-capstone-system-design.md`
- Create: `docs/phase-6/diagram/capstone/capstone-architecture.svg`

**Interfaces:**
- Consumes: prior phase capabilities documented in `docs/phase-2` through `docs/phase-5`.
- Produces: Phase6 learning route and implementation order.

- [x] **Step 1: Create architecture diagram**

Show Next.js UI, FastAPI backend, LangGraph Agent, knowledge layer, MCP tools, memory, observability, and evaluation.

- [x] **Step 2: Create Phase6 README**

Define subproject order:

```text
01-backend-skeleton
02-knowledge-ingestion
03-agentic-qa-runtime
04-web-ui
05-release-eval
```

- [x] **Step 3: Create design article**

Explain scope, boundaries, architecture, evaluation criteria, and why the first implementation step is backend skeleton.
