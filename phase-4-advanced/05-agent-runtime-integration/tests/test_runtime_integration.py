import sys
import tempfile
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[3]
RUNTIME_ROOT = CURRENT_DIR.parents[1]

sys.path.insert(0, str(RUNTIME_ROOT))

from project_tools import ProjectToolset
from runtime import IntegratedAgentRuntime


class RuntimeIntegrationTests(unittest.TestCase):
    def test_project_tools_search_docs_and_code_examples(self) -> None:
        tools = ProjectToolset(project_root=PROJECT_ROOT)

        docs = tools.search_docs("Agent Memory", phase="phase-4", limit=5)
        code = tools.find_code_examples("MemoryPolicy", phase="phase-4", limit=5)

        self.assertGreater(docs.count, 0)
        self.assertTrue(any(hit.path.endswith("03-agent-memory-system.md") for hit in docs.results))
        self.assertGreater(code.count, 0)
        self.assertTrue(any(hit.path.endswith("memory_policy.py") for hit in code.results))

    def test_project_tools_reject_empty_query(self) -> None:
        tools = ProjectToolset(project_root=PROJECT_ROOT)

        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            tools.search_docs("   ")

    def test_runtime_reads_memory_calls_tools_and_reviews_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime = IntegratedAgentRuntime(
                project_root=PROJECT_ROOT,
                memory_path=Path(tmp_dir) / "memory.json",
            )

            runtime.answer("以后回答我问题时，代码示例尽量用 Python。")
            result = runtime.answer("请结合 Phase4 Memory 的代码、文章和测试证据，说明是否可以进入 Phase5")

            self.assertTrue(any("Python" in memory.content for memory in result.memory_context))
            self.assertTrue(any(item.tool_name == "search_docs" for item in result.tool_results))
            self.assertTrue(any(item.tool_name == "find_code_examples" for item in result.tool_results))
            self.assertTrue(any(item.tool_name == "read_benchmark_summary" for item in result.tool_results))
            self.assertTrue(any(path.endswith("03-agent-memory-system.md") for path in result.evidence))
            self.assertTrue(any(path.endswith("memory_policy.py") for path in result.evidence))
            self.assertEqual(result.review.status.value, "approved")
            self.assertIn("supervisor.plan", result.trace)
            self.assertIn("memory.search", result.trace)
            self.assertIn("reviewer.review", result.trace)
            self.assertIn("风险", result.answer)

    def test_runtime_writes_explicit_task_memory_before_answering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime = IntegratedAgentRuntime(
                project_root=PROJECT_ROOT,
                memory_path=Path(tmp_dir) / "memory.json",
            )

            write_result = runtime.answer("记住：Phase4 当前任务是准备 Phase5 Production。")
            follow_up = runtime.answer("Phase4 当前任务是什么？")

            self.assertIsNotNone(write_result.written_memory)
            self.assertIn("memory.upsert", write_result.trace)
            self.assertTrue(any("Phase5 Production" in memory.content for memory in follow_up.memory_context))


if __name__ == "__main__":
    unittest.main()
