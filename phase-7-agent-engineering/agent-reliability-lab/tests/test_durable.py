from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_lab.durable import (
    DurableLoop,
    FaultPlan,
    IdempotencyConflictError,
    JsonRunStore,
    ScriptedModelService,
    TicketEffectStore,
    load_durable_cases,
    run_durable_eval,
)


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "datasets" / "durable-cases.jsonl"


class DurableLoopTests(unittest.TestCase):
    def test_dataset_covers_nine_fault_boundaries(self) -> None:
        cases = load_durable_cases(CASES_PATH)

        self.assertEqual(9, len(cases))
        self.assertEqual(
            {
                "clean-run",
                "restart-after-model",
                "model-transient-retry",
                "model-permanent-error",
                "retry-budget-exhausted",
                "write-receipt-recovery",
                "write-unknown",
                "cancel-at-human-wait",
                "stale-worker",
            },
            {case.case_id for case in cases},
        )

    def test_candidate_passes_release_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_durable_eval(CASES_PATH, Path(temporary))

        self.assertTrue(result.gate_passed)
        self.assertEqual(1.0, result.candidate.case_pass_rate)
        self.assertLess(result.baseline.case_pass_rate, 1.0)
        self.assertEqual(0, result.candidate.duplicate_side_effects)
        self.assertEqual(0, result.candidate.blind_retries)
        self.assertFalse(result.regressions)

    def test_committed_unknown_write_recovers_receipt_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_durable_eval(CASES_PATH, Path(temporary))
        run = _candidate(result, "write-receipt-recovery")

        self.assertEqual("completed", run.state.status)
        self.assertEqual(1, run.side_effect_count)
        self.assertEqual(0, run.duplicate_side_effects)
        self.assertIn(
            "receipt_recovered",
            [event["type"] for event in run.state.events],
        )

    def test_unconfirmed_write_waits_for_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_durable_eval(CASES_PATH, Path(temporary))
        run = _candidate(result, "write-unknown")

        self.assertEqual("waiting_reconciliation", run.state.status)
        self.assertEqual("result_unknown", run.state.failure_code)
        self.assertEqual(0, run.side_effect_count)

    def test_cancel_is_persisted_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_durable_eval(CASES_PATH, root)
            run = _candidate(result, "cancel-at-human-wait")
            stored = JsonRunStore(root / "cancel-at-human-wait").load(
                run.state.run_id
            )

        self.assertTrue(stored.cancel_requested)
        self.assertEqual("cancelled", stored.status)
        self.assertEqual(0, run.side_effect_count)

    def test_stale_worker_cannot_write_after_lease_takeover(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_durable_eval(CASES_PATH, Path(temporary))
        run = _candidate(result, "stale-worker")

        self.assertEqual("completed", run.state.status)
        self.assertEqual(1, run.side_effect_count)
        self.assertIn(
            "stale_worker_rejected",
            [event["type"] for event in run.state.events],
        )

    def test_permanent_error_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_durable_eval(CASES_PATH, Path(temporary))
        run = _candidate(result, "model-permanent-error")

        self.assertEqual(1, run.state.attempts["model"])
        self.assertEqual("invalid_request", run.state.failure_code)

    def test_same_action_id_with_new_arguments_is_rejected(self) -> None:
        effects = TicketEffectStore(FaultPlan("none"))
        effects.activate_fence(1)
        effects.record_followup(
            action_id="stable-action",
            logical_operation="followup::T-102",
            payload={"ticket_id": "T-102", "note": "first"},
            fence=1,
        )

        with self.assertRaises(IdempotencyConflictError):
            effects.record_followup(
                action_id="stable-action",
                logical_operation="followup::T-102",
                payload={"ticket_id": "T-102", "note": "changed"},
                fence=1,
            )

        self.assertEqual(1, len(effects.effects))

    def test_reconciliation_state_needs_explicit_recovery_action(self) -> None:
        case = next(
            case
            for case in load_durable_cases(CASES_PATH)
            if case.case_id == "write-unknown"
        )
        with tempfile.TemporaryDirectory() as temporary:
            faults = FaultPlan(case.fault)
            loop = DurableLoop(
                run_store=JsonRunStore(Path(temporary)),
                model=ScriptedModelService(faults),
                effects=TicketEffectStore(faults),
            )
            state = loop.start(case, worker_id="worker-a")

            self.assertEqual("waiting_reconciliation", state.status)
            with self.assertRaises(ValueError):
                loop.resume(case, worker_id="worker-b")


def _candidate(result, case_id: str):
    return next(
        run
        for run in result.runs
        if run.strategy == "durable-loop-v1" and run.case.case_id == case_id
    )


if __name__ == "__main__":
    unittest.main()
