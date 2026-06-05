# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

AI Agent learning project organized into 6 progressive phases, from fundamentals to a capstone enterprise knowledge-base Q&A system. Each phase directory contains independent sub-projects with their own dependencies.

## Tech Stack

- Python (Agent core logic, RAG, LangGraph/LangChain, FastAPI)
- TypeScript (Web UI with Next.js/React, MCP Servers)
- Key frameworks: LangGraph, LangChain, CrewAI, Codex Agent SDK
- Vector databases: Chroma (dev), Milvus (production)
- Observability: Langfuse
- Evaluation: RAGAS

## Architecture

Six-phase learning progression where each phase builds on the previous:

- `phase-1-fundamentals/` — Pure Python Agent implementations (ReAct, tool calling, memory)
- `phase-2-rag/` — RAG pipeline: chunking, hybrid search, reranking, RAGAS evaluation
- `phase-3-frameworks/` — LangChain, LangGraph, CrewAI, Codex Agent SDK implementations
- `phase-4-advanced/` — Memory systems, multi-agent patterns, MCP servers, security
- `phase-5-production/` — FastAPI backend, Docker deployment, observability, testing
- `phase-6-capstone/` — Enterprise knowledge-base Q&A Agent system (Next.js + FastAPI + LangGraph + Milvus)

Each sub-project (e.g., `phase-2-rag/01-basic-rag/`) is independently runnable with its own `requirements.txt` or `package.json`.

`docs/phase-{1..6}/` contains learning notes and article drafts.

## Working with Sub-Projects

Python projects: use `pip install -r requirements.txt` or `uv pip install -r requirements.txt` within each sub-project directory.

TypeScript projects: use `npm install` or `pnpm install` within each sub-project directory.

No root-level build system — navigate to the specific sub-project before running commands.

## Model Usage

- Planning and architecture: use Opus 4.7 (`Codex-opus-4-7`)
- Task execution and coding: use Sonnet 4.6 (`Codex-sonnet-4-6`)

## Conventions

- Python code follows standard patterns (type hints, docstrings where non-obvious)
- TypeScript follows standard Next.js/React conventions
- Each sub-project should be self-contained and runnable independently
- Articles go in `docs/phase-N/` as Markdown files
- Git tags mark phase completion (e.g., `phase-1-complete`)

## Technical Article / WeChat Writing Standards

Phase articles are not just learning notes. Treat publishable articles as engineering write-ups grounded in this repository's real code, traces, tests, and benchmark results.

- Open with a concrete question or tension, not a generic technology introduction. Good examples: "MCP 不就是把 API 包一层吗？", "Agent 记忆不就是 RAG 吗？", "Multi-Agent 不就是多几个角色聊天吗？"
- Use a progressive reader path: first clarify concept boundaries, then show the architecture, then walk through code, then discuss trade-offs and failure cases.
- Do not write interview-style Q&A as the final form unless explicitly requested. Interview questions can be used as hooks, but the article must still explain how the system is designed and implemented.
- Every core concept should map back to a local implementation file. Prefer references to actual paths under `phase-*` and `docs/phase-*` over abstract descriptions.
- Keep the engineering evidence visible: runnable demo commands, test output, benchmark numbers, graph traces, cost/latency data, and at least one limitation or failure case where applicable.
- Add enough visual structure for WeChat reading. For substantial articles, include at least three kinds of visuals when possible: a concept boundary diagram, a code architecture diagram, and an execution flow diagram.
- Use comparison tables for concepts that are easy to confuse, such as Chain vs Agent, Workflow vs Agent, Function Calling vs MCP vs Skills, RAG vs Memory, single-agent vs multi-agent.
- Maintain the project's own voice: real engineering reflection, code-backed conclusions, and honest trade-offs. Avoid turning articles into broad concept checklists or copied interview notes.
- End by connecting the topic back to the learning project: what this phase proves, what it does not prove yet, and how it prepares for the next phase or capstone.

## Skill Usage

This project only needs these skills. Do NOT invoke unrelated language/framework skills:

**Relevant:** coding-standards, frontend-patterns, backend-patterns, api-design, Codex-api, mcp-server-patterns, python-patterns, python-testing, tdd-workflow, e2e-testing, verification-loop, security-review, security-scan, pdf, pdf-extraction, waza-read, waza-write, waza-learn, strategic-compact, iterative-retrieval, eval-harness, continuous-learning, continuous-learning-v2, ai-regression-testing

**Irrelevant (do not use):** android-*, compose-*, cpp-*, csharp-*, dart-*, django-*, dotnet-*, golang-*, java-*, kotlin-*, laravel-*, nestjs-*, perl-*, rust-*, springboot-*, excel-*, ppt-*, frontend-slides, agent-browser, browser-controller, browser-use, chrome-devtools, x-api, api-harvester, article-writing, plankton-code-quality, configure-ecc, skill-stocktake, project-guidelines-example
