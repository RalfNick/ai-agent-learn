from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
CAPSTONE_ROOT = RUNTIME_ROOT.parents[0]
INGESTION_ROOT = CAPSTONE_ROOT / "02-knowledge-ingestion"
sys.path.insert(0, str(RUNTIME_ROOT))
sys.path.insert(0, str(INGESTION_ROOT))

from agentic_qa import AgenticQARuntime, build_runtime_from_sources
from knowledge import build_index_from_paths


class AgenticQARuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "agentic-rag.md").write_text(
            "\n".join(
                [
                    "# Agentic RAG Runtime",
                    "",
                    "Phase6 的 Agentic QA runtime 会保留 trace、sources 和 review_status。",
                    "当资料足够时，它应该基于证据回答。",
                    "当资料不足时，它应该拒答而不是编造。",
                ]
            ),
            encoding="utf-8",
        )
        (self.root / "frontend.md").write_text(
            "\n".join(
                [
                    "# Web UI",
                    "",
                    "Web UI 展示 chat、sources、trace view 和 eval summary。",
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_grounded_question_returns_sources_and_trace(self) -> None:
        index = build_index_from_paths([self.root], max_chars=180, overlap_chars=30)
        runtime = AgenticQARuntime(index=index, min_context_score=0.2, top_k=3)

        response = runtime.answer("Phase6 Agentic QA runtime 为什么需要 trace？", "session-a")

        self.assertEqual("agentic_rag", response.mode)
        self.assertEqual("session-a", response.session_id)
        self.assertEqual("evidence_supported", response.review_status)
        self.assertGreaterEqual(len(response.sources), 1)
        self.assertEqual("Agentic RAG Runtime", response.sources[0].title)
        self.assertIn("trace", response.answer)
        self.assertIn("sources", response.answer)
        self.assertIn("retrieve", [step.step for step in response.trace])
        self.assertIn("context_grade", [step.step for step in response.trace])
        self.assertIn("answer.generate", [step.step for step in response.trace])

    def test_unrelated_question_abstains(self) -> None:
        index = build_index_from_paths([self.root], max_chars=180, overlap_chars=30)
        runtime = AgenticQARuntime(index=index, min_context_score=0.85, top_k=2)

        response = runtime.answer("公司报销制度里的发票抬头是什么？", "session-b")

        self.assertEqual("agentic_rag", response.mode)
        self.assertEqual("abstained", response.review_status)
        self.assertEqual([], response.sources)
        self.assertIn("无法可靠回答", response.answer)
        self.assertEqual("abstain", response.trace[-1].step)

    def test_build_runtime_from_sources_loads_documents(self) -> None:
        runtime = build_runtime_from_sources([self.root], min_context_score=0.2, top_k=2)

        response = runtime.answer("Web UI 展示什么？")

        self.assertEqual("agentic_rag", response.mode)
        self.assertEqual("evidence_supported", response.review_status)
        self.assertEqual("Web UI", response.sources[0].title)

    def test_answer_generation_skips_markdown_tables_and_commands(self) -> None:
        (self.root / "trace-doc.md").write_text(
            "\n".join(
                [
                    "# Trace Design",
                    "",
                    "![Trace Diagram](trace.svg)",
                    "返回 agentic_rag、sources、trace、review_status",
                    "",
                    "| 能力 | 为什么需要 |",
                    "| --- | --- |",
                    "| trace | 开发者需要调试路由 |",
                    "",
                    "```bash",
                    "python3 run_agentic_qa.py --question \"为什么需要 trace\"",
                    "返回 agentic_rag、sources、trace、review_status",
                    "```",
                    "",
                    "前端需要什么字段？",
                    "Trace 能展示每一步路由、检索来源、耗时和 review 结果。",
                ]
            ),
            encoding="utf-8",
        )
        runtime = build_runtime_from_sources([self.root], min_context_score=0.2, top_k=2)

        response = runtime.answer("为什么需要 trace？")

        self.assertIn("trace：开发者需要调试路由", response.answer)
        self.assertNotIn("| 能力 |", response.answer)
        self.assertNotIn("python3 run_agentic_qa.py", response.answer)
        self.assertNotIn("返回 agentic_rag", response.answer)
        self.assertNotIn("前端需要什么字段？", response.answer)

    def test_answer_generation_can_use_useful_table_rows(self) -> None:
        (self.root / "trace-table.md").write_text(
            "\n".join(
                [
                    "# Trace Table",
                    "",
                    "| 能力 | 为什么需要 |",
                    "| --- | --- |",
                    "| trace 展示 | 开发者要能调试路径 |",
                ]
            ),
            encoding="utf-8",
        )
        runtime = build_runtime_from_sources([self.root], min_context_score=0.2, top_k=1)

        response = runtime.answer("为什么需要 trace？")

        self.assertIn("trace 展示：开发者要能调试路径", response.answer)
        self.assertNotIn("| 能力 |", response.answer)

    def test_answer_generation_prioritizes_exact_technical_terms(self) -> None:
        (self.root / "trace-priority.md").write_text(
            "\n".join(
                [
                    "# Trace Priority",
                    "",
                    "| 能力 | 为什么需要 |",
                    "| --- | --- |",
                    "| LangGraph workflow | 需要 rewrite、repair、abstain |",
                    "| trace 展示 | 开发者要能调试路径 |",
                ]
            ),
            encoding="utf-8",
        )
        runtime = build_runtime_from_sources([self.root], min_context_score=0.2, top_k=1)

        response = runtime.answer("Phase6 为什么需要 trace？")

        self.assertIn("trace 展示：开发者要能调试路径", response.answer)
        self.assertNotIn("LangGraph workflow：需要 rewrite", response.answer)

    def test_langgraph_workflow_repairs_unsupported_answer_lines(self) -> None:
        index = build_index_from_paths([self.root], max_chars=180, overlap_chars=30)

        def unsafe_answer_builder(question, results):
            return "\n".join(
                [
                    "根据当前知识库资料，可以确认：",
                    "1. Phase6 的 Agentic QA runtime 会保留 trace。（来源：Agentic RAG Runtime）",
                    "2. 公司报销制度要求发票抬头固定为测试公司。",
                ]
            )

        runtime = AgenticQARuntime(
            index=index,
            min_context_score=0.2,
            top_k=2,
            unsafe_answer_builder=unsafe_answer_builder,
            max_repairs=1,
        )

        response = runtime.answer("Phase6 Agentic QA runtime 为什么需要 trace？", "repair-session")

        self.assertEqual("evidence_supported", response.review_status)
        self.assertNotIn("发票抬头", response.answer)
        steps = [step.step for step in response.trace]
        self.assertIn("review.failed", steps)
        self.assertIn("answer.repair", steps)


if __name__ == "__main__":
    unittest.main()
