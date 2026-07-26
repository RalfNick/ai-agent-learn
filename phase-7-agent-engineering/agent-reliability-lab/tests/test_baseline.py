from __future__ import annotations

import unittest
from pathlib import Path

from agent_lab.baseline import run_baseline


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


if __name__ == "__main__":
    unittest.main()
