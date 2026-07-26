from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from agent_lab.baseline import load_tasks, run_baseline


ROOT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    def test_baseline_passes_the_control_dataset(self) -> None:
        result = run_baseline(
            ROOT / "datasets" / "tasks.jsonl",
            ROOT / "fixtures" / "knowledge" / "product-handbook.md",
        )
        self.assertEqual(result.total, 5)
        self.assertEqual(result.passed, 5)
        self.assertEqual(result.task_pass_rate, 1.0)

    def test_unknown_internal_fact_is_not_answered(self) -> None:
        result = run_baseline(
            ROOT / "datasets" / "tasks.jsonl",
            ROOT / "fixtures" / "knowledge" / "product-handbook.md",
        )
        unknown_case = next(case for case in result.cases if case.task_id == "qa-005")
        self.assertEqual(unknown_case.status, "abstained")

    def test_zero_threshold_exposes_over_answering(self) -> None:
        result = run_baseline(
            ROOT / "datasets" / "tasks.jsonl",
            ROOT / "fixtures" / "knowledge" / "product-handbook.md",
            threshold=0.0,
        )
        unknown_case = next(case for case in result.cases if case.task_id == "qa-005")
        self.assertEqual(unknown_case.status, "answered")
        self.assertFalse(unknown_case.passed)

    def test_high_threshold_exposes_over_abstention(self) -> None:
        result = run_baseline(
            ROOT / "datasets" / "tasks.jsonl",
            ROOT / "fixtures" / "knowledge" / "product-handbook.md",
            threshold=1.0,
        )
        self.assertLess(result.task_pass_rate, 1.0)

    def test_duplicate_task_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text(
                '{"id":"qa-001","question":"A","expected_status":"abstained"}\n'
                '{"id":"qa-001","question":"B","expected_status":"abstained"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate task id"):
                load_tasks(path)


if __name__ == "__main__":
    unittest.main()
