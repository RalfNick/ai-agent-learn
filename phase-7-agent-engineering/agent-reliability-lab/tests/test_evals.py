from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_lab.evals import (
    SystemOutput,
    grade_output,
    load_eval_tasks,
    run_eval,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "datasets" / "eval-tasks.jsonl"
KNOWLEDGE = ROOT / "fixtures" / "knowledge" / "product-handbook.md"


class EvalTests(unittest.TestCase):
    def test_candidate_improves_without_regressions(self) -> None:
        result = run_eval(TASKS, KNOWLEDGE, trials_per_task=3)
        self.assertTrue(result.gate_passed)
        self.assertGreater(len(result.improvements), 0)
        self.assertEqual(result.regressions, [])
        self.assertEqual(result.unstable_candidate_tasks, [])

    def test_eval_runs_every_task_multiple_times(self) -> None:
        result = run_eval(TASKS, KNOWLEDGE, trials_per_task=3)
        task_count = len(load_eval_tasks(TASKS))
        self.assertEqual(len(result.trials), task_count * 3 * 2)
        self.assertEqual(result.baseline.trials, task_count * 3)
        self.assertEqual(result.candidate.trials, task_count * 3)

    def test_flaky_simulator_fails_the_release_gate(self) -> None:
        result = run_eval(
            TASKS,
            KNOWLEDGE,
            trials_per_task=3,
            candidate_id="flaky-simulator",
        )
        self.assertFalse(result.gate_passed)
        self.assertGreater(len(result.unstable_candidate_tasks), 0)

    def test_forbidden_term_grader_detects_unsafe_output(self) -> None:
        task = next(task for task in load_eval_tasks(TASKS) if task.task_id == "reg-003")
        output = SystemOutput(
            status="answered",
            answer="可以明文记录密钥。",
            score=1.0,
            source="敏感数据",
            trace_steps=["retrieve", "threshold_gate", "answer"],
            normalized_question=task.question,
            latency_ms=0.0,
        )
        grades = {grade.name: grade for grade in grade_output(task, output)}
        self.assertFalse(grades["forbidden_terms"].passed)
        self.assertFalse(grades["required_terms"].passed)

    def test_duplicate_eval_task_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text(
                '{"id":"same","suite":"regression","risk":"normal","question":"A",'
                '"expected_status":"abstained"}\n'
                '{"id":"same","suite":"regression","risk":"normal","question":"B",'
                '"expected_status":"abstained"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate task id"):
                load_eval_tasks(path)


if __name__ == "__main__":
    unittest.main()
