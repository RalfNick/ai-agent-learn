from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
CAPSTONE_ROOT = EVAL_ROOT.parents[0]
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(CAPSTONE_ROOT / "02-knowledge-ingestion"))
sys.path.insert(0, str(CAPSTONE_ROOT / "03-agentic-qa-runtime"))

from release_eval import EvalCase, evaluate_cases


class ReleaseEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "trace.md").write_text(
            "\n".join(
                [
                    "# Trace Evidence",
                    "",
                    "| 能力 | 为什么需要 |",
                    "| --- | --- |",
                    "| trace 展示 | 开发者要能调试路径 |",
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_evaluate_cases_reports_pass_rate(self) -> None:
        cases = [
            EvalCase(
                case_id="trace-pass",
                question="为什么需要 trace？",
                expected_terms=["trace 展示", "调试路径"],
                expected_review_status="evidence_supported",
                expected_source_title="Trace Evidence",
            ),
            EvalCase(
                case_id="trace-fail",
                question="为什么需要 trace？",
                expected_terms=["不存在的结论"],
                expected_review_status="evidence_supported",
                expected_source_title="Trace Evidence",
            ),
        ]

        summary = evaluate_cases(cases=cases, source_paths=[self.root])

        self.assertEqual(2, summary.total)
        self.assertEqual(1, summary.passed)
        self.assertEqual(0.5, summary.pass_rate)
        self.assertTrue(summary.records[0].passed)
        self.assertFalse(summary.records[1].passed)
        self.assertIn("missing_terms", summary.records[1].failures[0])

    def test_evaluate_cases_checks_routes_negative_terms_and_wrong_sources(self) -> None:
        cases = [
            EvalCase(
                case_id="weak-context-abstain",
                question="ZYXW-999 这个不存在的系统是什么？",
                expected_terms=["无法可靠回答"],
                expected_review_status="abstained",
                expected_trace_steps=["context_grade", "abstain"],
                min_context_score=0.99,
            ),
            EvalCase(
                case_id="repair-unsafe-answer",
                question="为什么需要 trace？",
                expected_terms=["trace 展示"],
                forbidden_terms=["发票抬头"],
                expected_review_status="evidence_supported",
                expected_source_title="Trace Evidence",
                expected_trace_steps=["review.failed", "answer.repair", "review.evidence_supported"],
                force_unsafe_answer=True,
            ),
            EvalCase(
                case_id="wrong-source-fails",
                question="为什么需要 trace？",
                expected_terms=["trace 展示"],
                expected_review_status="evidence_supported",
                expected_source_title="Wrong Source",
            ),
        ]

        summary = evaluate_cases(cases=cases, source_paths=[self.root])

        self.assertEqual(3, summary.total)
        self.assertEqual(2, summary.passed)
        self.assertTrue(summary.records[0].passed)
        self.assertTrue(summary.records[1].passed)
        self.assertFalse(summary.records[2].passed)
        self.assertIn("missing_source:Wrong Source", summary.records[2].failures)


if __name__ == "__main__":
    unittest.main()
