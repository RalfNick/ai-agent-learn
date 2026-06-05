import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import AgentRole, ReviewStatus
from handoff import HandoffPacket
from supervisor import MultiAgentSupervisor, ReviewerAgent


class MultiAgentPatternTests(unittest.TestCase):
    def test_supervisor_routes_article_and_code_questions_to_specialists(self) -> None:
        supervisor = MultiAgentSupervisor()

        plan = supervisor.plan("Review Phase4 Memory 的代码和文章，指出下一步怎么优化")

        self.assertEqual(
            [packet.target for packet in plan.handoffs],
            [AgentRole.DOC_RESEARCHER, AgentRole.CODE_ANALYST],
        )
        self.assertTrue(all(packet.required_outputs for packet in plan.handoffs))

    def test_handoff_packet_is_an_explicit_contract(self) -> None:
        packet = HandoffPacket(
            target=AgentRole.CODE_ANALYST,
            task="检查 MemoryPolicy 是否有边界问题",
            context={"phase": "phase-4", "component": "memory"},
            required_outputs=["file_refs", "risks"],
            constraints=["不要修改文件"],
        )

        serialized = packet.to_dict()

        self.assertEqual(serialized["target"], "code_analyst")
        self.assertEqual(serialized["context"]["component"], "memory")
        self.assertIn("risks", serialized["required_outputs"])
        self.assertIn("不要修改文件", serialized["constraints"])

    def test_reviewer_rejects_answers_without_evidence(self) -> None:
        reviewer = ReviewerAgent()

        rejected = reviewer.review("结论：系统已经足够好了。", evidence=[])
        approved = reviewer.review(
            "结论：MemoryPolicy 已经覆盖敏感词和中文项目名。",
            evidence=[
                "phase-4-advanced/04-multi-agent-patterns/tests/test_multi_agent_patterns.py",
                "phase-4-advanced/03-memory-system/memory_policy.py",
            ],
        )

        self.assertEqual(rejected.status, ReviewStatus.NEEDS_EVIDENCE)
        self.assertEqual(approved.status, ReviewStatus.APPROVED)
        self.assertGreater(approved.score, rejected.score)

    def test_supervisor_run_returns_trace_evidence_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            supervisor = MultiAgentSupervisor(project_root=Path(tmp_dir))

            result = supervisor.run("请评估 Phase4 Memory 的代码、文章和 benchmark 证据")

            self.assertIn("supervisor.plan", result.trace)
            self.assertIn("reviewer.review", result.trace)
            self.assertGreaterEqual(len(result.handoffs), 3)
            self.assertTrue(result.evidence)
            self.assertEqual(result.review.status, ReviewStatus.APPROVED)
            self.assertIn("DocResearchAgent", result.answer)
            self.assertIn("CodeAnalysisAgent", result.answer)
            self.assertIn("BenchmarkAgent", result.answer)


if __name__ == "__main__":
    unittest.main()
