# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Agent learning project organized into 6 progressive phases, from fundamentals to a capstone enterprise knowledge-base Q&A system. Each phase directory contains independent sub-projects with their own dependencies.

## Tech Stack

- Python (Agent core logic, RAG, LangGraph/LangChain, FastAPI)
- TypeScript (Web UI with Next.js/React, MCP Servers)
- Key frameworks: LangGraph, LangChain, CrewAI, Claude Agent SDK
- Vector databases: Chroma (dev), Milvus (production)
- Observability: Langfuse
- Evaluation: RAGAS

## Architecture

Six-phase learning progression where each phase builds on the previous:

- `phase-1-fundamentals/` — Pure Python Agent implementations (ReAct, tool calling, memory)
- `phase-2-rag/` — RAG pipeline: chunking, hybrid search, reranking, RAGAS evaluation
- `phase-3-frameworks/` — LangChain, LangGraph, CrewAI, Claude Agent SDK implementations
- `phase-4-advanced/` — Memory systems, multi-agent patterns, MCP servers, security
- `phase-5-production/` — FastAPI backend, Docker deployment, observability, testing
- `phase-6-capstone/` — Enterprise knowledge-base Q&A Agent system (Next.js + FastAPI + LangGraph + Milvus)

Each sub-project (e.g., `phase-2-rag/01-basic-rag/`) is independently runnable with its own `requirements.txt` or `package.json`.

`docs/phase-{1..6}/` contains learning notes and article drafts.

## Working with Sub-Projects

Python projects: use `pip install -r requirements.txt` or `uv pip install -r requirements.txt` within each sub-project directory.

TypeScript projects: use `npm install` or `pnpm install` within each sub-project directory.

No root-level build system — navigate to the specific sub-project before running commands.

## Conventions

- Python code follows standard patterns (type hints, docstrings where non-obvious)
- TypeScript follows standard Next.js/React conventions
- Each sub-project should be self-contained and runnable independently
- Articles go in `docs/phase-N/` as Markdown files
- Git tags mark phase completion (e.g., `phase-1-complete`)

## Skill Usage

This project only needs these skills. Do NOT invoke unrelated language/framework skills:

**Relevant:** coding-standards, frontend-patterns, backend-patterns, api-design, claude-api, mcp-server-patterns, python-patterns, python-testing, tdd-workflow, e2e-testing, verification-loop, security-review, security-scan, pdf, pdf-extraction, waza-read, waza-write, waza-learn, strategic-compact, iterative-retrieval, eval-harness, continuous-learning, continuous-learning-v2, ai-regression-testing

**Irrelevant (do not use):** android-*, compose-*, cpp-*, csharp-*, dart-*, django-*, dotnet-*, golang-*, java-*, kotlin-*, laravel-*, nestjs-*, perl-*, rust-*, springboot-*, excel-*, ppt-*, frontend-slides, agent-browser, browser-controller, browser-use, chrome-devtools, x-api, api-harvester, article-writing, plankton-code-quality, configure-ecc, skill-stocktake, project-guidelines-example
