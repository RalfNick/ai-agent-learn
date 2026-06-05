# Phase4 Runtime Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small deterministic runtime that integrates Phase4 MCP-style tools, long-term memory, and multi-agent review into one runnable flow.

**Architecture:** Keep the integration layer separate under `phase-4-advanced/05-agent-runtime-integration/`. Reuse existing Phase4 Memory and Multi-Agent modules through explicit import paths, and implement Python read-only project tools that mirror the MCP Server's three tool concepts.

**Tech Stack:** Python standard library, `unittest`, existing Phase4 Python modules.

---

### Task 1: Define Runtime Behavior With Tests

**Files:**
- Create: `phase-4-advanced/05-agent-runtime-integration/tests/test_runtime_integration.py`

- [ ] Write tests that prove the runtime recalls long-term memory, calls read-only project tools, routes through multi-agent planning, returns evidence, and rejects empty tool queries.
- [ ] Run the new tests and confirm they fail because runtime modules do not exist yet.

### Task 2: Implement Read-Only Project Tools

**Files:**
- Create: `phase-4-advanced/05-agent-runtime-integration/project_tools.py`

- [ ] Implement `ProjectToolset.search_docs`, `find_code_examples`, and `read_benchmark_summary`.
- [ ] Keep all tools read-only and scoped to `docs/` plus phase directories.
- [ ] Run the new tests and confirm tool tests pass.

### Task 3: Implement Integrated Runtime

**Files:**
- Create: `phase-4-advanced/05-agent-runtime-integration/runtime.py`
- Create: `phase-4-advanced/05-agent-runtime-integration/runtime_demo.py`
- Create: `phase-4-advanced/05-agent-runtime-integration/README.md`

- [ ] Implement `IntegratedAgentRuntime.answer(question)`.
- [ ] Read relevant memory before planning, write explicit memory through `MemoryPolicy`, call project tools according to supervisor handoffs, review evidence, and return a trace.
- [ ] Add a runnable demo that prints answer, memory, tool evidence, review, and trace.

### Task 4: Write Phase4 Closure Article

**Files:**
- Create: `docs/phase-4/05-agent-runtime-integration.md`
- Modify: `docs/phase-4/README.md`
- Modify: `phase-4-advanced/README.md`

- [ ] Write a publishable technical article with a concrete opening question, architecture diagram, code map, trace, tests, trade-offs, and next-step connection to Phase5.
- [ ] Update Phase4 indexes so the integration step appears after MCP, Memory, and Multi-Agent.

### Task 5: Verify

- [ ] Run `python3 -m unittest discover -s phase-4-advanced/05-agent-runtime-integration/tests`.
- [ ] Run existing Phase4 Memory and Multi-Agent tests.
- [ ] Run `python3 phase-4-advanced/05-agent-runtime-integration/runtime_demo.py`.
- [ ] Validate article image paths and Markdown H1 count.
