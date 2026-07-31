from __future__ import annotations

import unittest
from pathlib import Path

from agent_lab.harness import (
    MinimalHarness,
    RunState,
    load_harness_cases,
    run_harness_eval,
)


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "datasets" / "harness-cases.jsonl"


class HarnessEvalTests(unittest.TestCase):
    def test_dataset_has_six_boundary_cases(self) -> None:
        cases = load_harness_cases(CASES_PATH)

        self.assertEqual(6, len(cases))
        self.assertEqual(
            {
                "read-only-answer",
                "approval-pause",
                "approval-resume",
                "tool-timeout",
                "step-budget",
                "verification-failure",
            },
            {case.case_id for case in cases},
        )

    def test_candidate_passes_all_cases(self) -> None:
        result = run_harness_eval(CASES_PATH)

        self.assertTrue(result.gate_passed)
        self.assertEqual(1.0, result.candidate.case_pass_rate)
        self.assertLess(result.baseline.case_pass_rate, 1.0)
        self.assertFalse(result.regressions)

    def test_pause_serializes_before_any_write(self) -> None:
        case = _case("approval-pause")
        harness = MinimalHarness(max_steps=3, tool_timeout_ms=500)

        state = harness.start(case)
        restored = RunState.from_json(state.to_json())
        types = [event["type"] for event in restored.events]

        self.assertEqual("waiting_approval", restored.status)
        self.assertEqual([], harness.tools.side_effects)
        self.assertLess(
            types.index("checkpoint_saved"),
            types.index("approval_requested"),
        )

    def test_approved_resume_writes_once(self) -> None:
        case = _case("approval-resume")
        harness = MinimalHarness(max_steps=3, tool_timeout_ms=500)

        paused = harness.start(case)
        completed = harness.resume(case, paused, approve=True)

        self.assertEqual("completed", completed.status)
        self.assertEqual(1, len(harness.tools.side_effects))
        self.assertIn("write-followup-102", completed.completed_action_ids)
        self.assertIn(
            "run_resumed",
            [event["type"] for event in completed.events],
        )

    def test_timeout_and_step_budget_are_explicit(self) -> None:
        timeout_harness = MinimalHarness(max_steps=3, tool_timeout_ms=500)
        timeout_state = timeout_harness.start(_case("tool-timeout"))
        loop_harness = MinimalHarness(max_steps=3, tool_timeout_ms=500)
        loop_state = loop_harness.start(_case("step-budget"))

        self.assertEqual("tool_timeout", timeout_state.failure_code)
        self.assertEqual("failed", timeout_state.status)
        self.assertEqual("max_steps", loop_state.failure_code)
        self.assertEqual("stopped", loop_state.status)

    def test_verifier_failure_is_not_reported_as_success(self) -> None:
        harness = MinimalHarness(max_steps=3, tool_timeout_ms=500)

        state = harness.start(_case("verification-failure"))

        self.assertEqual("failed_verification", state.status)
        self.assertEqual("missing_evidence", state.failure_code)

    def test_tighter_step_override_fails_release_gate(self) -> None:
        result = run_harness_eval(CASES_PATH, max_steps=1)

        self.assertFalse(result.gate_passed)
        self.assertLess(result.candidate.case_pass_rate, 1.0)


def _case(case_id: str):
    return next(
        case for case in load_harness_cases(CASES_PATH)
        if case.case_id == case_id
    )


if __name__ == "__main__":
    unittest.main()
