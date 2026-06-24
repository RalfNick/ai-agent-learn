# Phase6 Knowledge Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, testable knowledge ingestion slice for the Phase6 capstone that can load repository documents, split them into chunks, persist an index, and retrieve relevant chunks for later Agentic QA.

**Architecture:** `02-knowledge-ingestion` is a standalone Python sub-project. It exposes focused modules for loading, chunking, indexing, retrieval, and two CLIs so the next slice can reuse the same retrieval contract without depending on FastAPI internals.

**Tech Stack:** Python 3.11+, stdlib JSON/pathlib/argparse/math, optional `pypdf` for PDF text extraction, `unittest` for tests.

## Global Constraints

- Keep the slice independently runnable from `phase-6-capstone/02-knowledge-ingestion/`.
- Do not call an LLM or external embedding API in this slice.
- Persist index data as local JSON so tests and demos are deterministic.
- Prefer a clear retrieval interface over production vector database complexity.
- Treat PDF support as optional and gracefully report missing parser dependencies.

---

### Task 1: Tests For Loading, Chunking, And Retrieval

**Files:**
- Create: `phase-6-capstone/02-knowledge-ingestion/tests/test_knowledge_ingestion.py`

**Interfaces:**
- Consumes: none.
- Produces expectations for `load_documents`, `MarkdownChunker`, `LocalKnowledgeIndex`, `build_index_from_paths`.

- [x] **Step 1: Write failing tests**

Tests should assert:
- Markdown files are loaded with `path`, `title`, `content`, and `document_id`.
- Chunking preserves source metadata and produces non-empty chunks.
- Hybrid retrieval returns the most relevant chunk for a query.
- Saved index can be reloaded without changing search results.

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/02-knowledge-ingestion/tests`

Expected: FAIL because `knowledge` package does not exist.

### Task 2: Minimal Knowledge Package

**Files:**
- Create: `phase-6-capstone/02-knowledge-ingestion/knowledge/__init__.py`
- Create: `phase-6-capstone/02-knowledge-ingestion/knowledge/models.py`
- Create: `phase-6-capstone/02-knowledge-ingestion/knowledge/loaders.py`
- Create: `phase-6-capstone/02-knowledge-ingestion/knowledge/chunking.py`
- Create: `phase-6-capstone/02-knowledge-ingestion/knowledge/index.py`

**Interfaces:**
- `load_documents(paths: Sequence[Path | str], extensions: set[str] | None = None) -> list[Document]`
- `MarkdownChunker(max_chars: int = 900, overlap_chars: int = 120).split_documents(documents: Sequence[Document]) -> list[Chunk]`
- `LocalKnowledgeIndex.from_chunks(chunks: Sequence[Chunk]) -> LocalKnowledgeIndex`
- `LocalKnowledgeIndex.search(query: str, limit: int = 5) -> list[RetrievalResult]`
- `LocalKnowledgeIndex.save(path: Path | str) -> None`
- `LocalKnowledgeIndex.load(path: Path | str) -> LocalKnowledgeIndex`
- `build_index_from_paths(paths: Sequence[Path | str], max_chars: int, overlap_chars: int, extensions: set[str] | None = None) -> LocalKnowledgeIndex`

- [x] **Step 1: Implement only the behavior required by tests**

Use deterministic tokenization and a simple hybrid score:
- lexical score from token overlap.
- vector score from hashed term-frequency cosine similarity.
- final score = `0.65 * lexical + 0.35 * vector`.

- [x] **Step 2: Run tests to verify green**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/02-knowledge-ingestion/tests`

Expected: `Ran 5 tests OK`.

### Task 3: CLI And Documentation

**Files:**
- Create: `phase-6-capstone/02-knowledge-ingestion/ingest.py`
- Create: `phase-6-capstone/02-knowledge-ingestion/search.py`
- Create: `phase-6-capstone/02-knowledge-ingestion/requirements.txt`
- Create: `phase-6-capstone/02-knowledge-ingestion/README.md`
- Create: `docs/phase-6/02-knowledge-ingestion.md`
- Modify: `phase-6-capstone/README.md`
- Modify: `docs/phase-6/README.md`

**Interfaces:**
- `python3 ingest.py --source ../../docs/phase-6 --index .local/phase6-index.json`
- `python3 search.py --index .local/phase6-index.json --query "Agentic RAG trace"`

- [x] **Step 1: Add CLIs**

`ingest.py` prints JSON stats with document count, chunk count, and index path.

`search.py` prints ranked JSON results with title, path, score, and snippet.

- [x] **Step 2: Add docs**

Document the current limits: local deterministic retrieval, no rerank yet, PDF parser optional, no API integration yet.

### Task 4: Verification

**Files:**
- No new files.

- [x] **Step 1: Run focused tests**

`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/02-knowledge-ingestion/tests`

- [x] **Step 2: Run py_compile**

`PYTHONPYCACHEPREFIX=/private/tmp/ai-agent-learn-pycache python3 -m py_compile phase-6-capstone/02-knowledge-ingestion/knowledge/*.py phase-6-capstone/02-knowledge-ingestion/*.py`

- [x] **Step 3: Run CLI smoke**

`python3 phase-6-capstone/02-knowledge-ingestion/ingest.py --source docs/phase-6 --index /tmp/phase6-knowledge-index.json`

`python3 phase-6-capstone/02-knowledge-ingestion/search.py --index /tmp/phase6-knowledge-index.json --query "企业知识库 Agent trace"`
