from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

INGESTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INGESTION_ROOT))

from knowledge import (
    LocalKnowledgeIndex,
    MarkdownChunker,
    build_index_from_paths,
    load_documents,
)


class KnowledgeIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.agent_doc = self.root / "agentic-rag.md"
        self.deploy_doc = self.root / "docker-deploy.md"
        self.memory_doc = self.root / "memory-system.md"
        self.agent_doc.write_text(
            "\n".join(
                [
                    "# Agentic RAG",
                    "",
                    "Agentic RAG uses graph trace, faithfulness review,",
                    "query rewrite, answer repair, and abstain routing.",
                    "",
                    "## Runtime",
                    "",
                    "The runtime should preserve sources and evidence.",
                ]
            ),
            encoding="utf-8",
        )
        self.deploy_doc.write_text(
            "\n".join(
                [
                    "# Docker Deploy",
                    "",
                    "Docker Compose starts the FastAPI service with healthcheck",
                    "and environment variables for deployment.",
                ]
            ),
            encoding="utf-8",
        )
        self.memory_doc.write_text(
            "\n".join(
                [
                    "# 企业知识库记忆系统",
                    "",
                    "这个系统需要支持长期记忆、知识检索、会话状态和证据引用。",
                    "Agent 在资料不足时应该拒答。",
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_documents_reads_markdown_metadata(self) -> None:
        documents = load_documents([self.root])

        titles = {document.title for document in documents}
        self.assertEqual({"Agentic RAG", "Docker Deploy", "企业知识库记忆系统"}, titles)
        first_doc = next(document for document in documents if document.title == "Agentic RAG")
        self.assertEqual(str(self.agent_doc.resolve()), first_doc.path)
        self.assertTrue(first_doc.document_id)
        self.assertIn("faithfulness review", first_doc.content)

    def test_chunker_preserves_source_metadata(self) -> None:
        documents = load_documents([self.agent_doc])
        chunks = MarkdownChunker(max_chars=90, overlap_chars=20).split_documents(documents)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.document_id == documents[0].document_id for chunk in chunks))
        self.assertTrue(all(chunk.path == str(self.agent_doc.resolve()) for chunk in chunks))
        self.assertTrue(all(chunk.content.strip() for chunk in chunks))
        self.assertIn("Agentic RAG", chunks[0].title)

    def test_hybrid_retrieval_returns_relevant_chunks(self) -> None:
        index = build_index_from_paths([self.root], max_chars=120, overlap_chars=20)

        results = index.search("faithfulness trace repair routing", limit=2)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual("Agentic RAG", results[0].chunk.title)
        self.assertGreater(results[0].score, 0)
        self.assertGreaterEqual(results[0].lexical_score, results[0].vector_score * 0)

    def test_hybrid_retrieval_handles_chinese_terms(self) -> None:
        index = build_index_from_paths([self.root], max_chars=120, overlap_chars=20)

        results = index.search("知识库 检索 证据", limit=1)

        self.assertEqual("企业知识库记忆系统", results[0].chunk.title)

    def test_saved_index_can_be_reloaded(self) -> None:
        index_path = self.root / "index.json"
        index = build_index_from_paths([self.root], max_chars=120, overlap_chars=20)
        index.save(index_path)

        raw = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertEqual(3, raw["stats"]["document_count"])

        loaded = LocalKnowledgeIndex.load(index_path)
        results = loaded.search("FastAPI healthcheck deployment", limit=1)

        self.assertEqual("Docker Deploy", results[0].chunk.title)
        self.assertEqual(index.stats().chunk_count, loaded.stats().chunk_count)


if __name__ == "__main__":
    unittest.main()
