from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent_lab.security import (
    ActionProposal,
    ApprovalDecision,
    JsonApprovalStore,
    SecurityEngine,
    action_fingerprint,
    load_security_cases,
    run_security_eval,
)
from agent_lab.security_reporting import write_security_reports


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "datasets" / "security-cases.jsonl"


class SecurityFixtureTests(unittest.TestCase):
    def test_dataset_covers_eight_control_boundaries(self) -> None:
        cases = load_security_cases(CASES_PATH)

        self.assertEqual(8, len(cases))
        self.assertEqual(
            {
                "read-only-auto",
                "reversible-policy-allowed",
                "external-approved-once",
                "external-rejected",
                "approval-expired",
                "arguments-changed-after-approval",
                "critical-wrong-reviewer",
                "duplicate-approved-resume",
            },
            {case.case_id for case in cases},
        )

    def test_action_fingerprint_is_canonical_and_argument_bound(self) -> None:
        first = _action(
            action_id="act-1",
            arguments={"ticket_id": "T-1", "message": "reviewed"},
        )
        reordered = _action(
            action_id="act-1",
            arguments={"message": "reviewed", "ticket_id": "T-1"},
        )
        changed = _action(
            action_id="act-1",
            arguments={"ticket_id": "T-1", "message": "changed"},
        )

        self.assertEqual(action_fingerprint(first), action_fingerprint(reordered))
        self.assertNotEqual(action_fingerprint(first), action_fingerprint(changed))


class SecurityEngineTests(unittest.TestCase):
    def test_read_and_bounded_reversible_actions_follow_policy(self) -> None:
        engine = SecurityEngine()
        read = engine.propose(
            _action(tool="lookup_ticket", arguments={"ticket_id": "T-1"}),
            now=100,
        )
        reversible = engine.propose(
            _action(
                action_id="act-2",
                tool="update_ticket_label",
                arguments={"ticket_id": "T-1", "label": "needs-review"},
                rollback={
                    "tool": "restore_ticket_labels",
                    "arguments": {"ticket_id": "T-1"},
                },
            ),
            now=100,
        )

        self.assertEqual(("completed", "auto_allowed"), (read.status, read.reason))
        self.assertEqual(
            ("completed", "policy_allowed"),
            (reversible.status, reversible.reason),
        )
        self.assertEqual(1, engine.effects.mutation_count)
        self.assertIsNotNone(reversible.rollback)

    def test_external_action_has_no_effect_before_approval(self) -> None:
        engine = SecurityEngine()
        state = engine.propose(_action(), now=100)

        self.assertEqual("waiting_approval", state.status)
        self.assertEqual("approval_required", state.reason)
        self.assertIsNotNone(state.approval)
        self.assertEqual(0, engine.effects.mutation_count)

    def test_pending_approval_survives_process_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = JsonApprovalStore(Path(temporary))
            first_engine = SecurityEngine()
            pending = first_engine.propose(_action(), now=100)
            store.save(pending)

            restored = store.load(pending.action.action_id)
            second_engine = SecurityEngine()
            completed = second_engine.resume(restored, _decision())

        self.assertEqual(("completed", "approved"), (completed.status, completed.reason))
        self.assertEqual(1, second_engine.effects.mutation_count)
        self.assertEqual(
            pending.approval.action_fingerprint,
            restored.approval.action_fingerprint,
        )

    def test_approved_resume_executes_once(self) -> None:
        engine = SecurityEngine()
        state = engine.propose(_action(), now=100)
        decision = _decision()

        completed = engine.resume(state, decision)
        replayed = engine.resume(completed, decision)

        self.assertEqual(("completed", "approved"), (replayed.status, replayed.reason))
        self.assertEqual(1, engine.effects.mutation_count)
        self.assertIn("receipt_replayed", [event["type"] for event in replayed.events])

    def test_rejection_expiry_and_argument_drift_never_execute(self) -> None:
        for expected_status, expected_reason, decision, current_action in (
            (
                "rejected",
                "rejected_by_reviewer",
                _decision(decision_type="reject"),
                None,
            ),
            (
                "expired",
                "approval_expired",
                _decision(at=121),
                None,
            ),
            (
                "blocked",
                "action_changed",
                _decision(),
                _action(arguments={"ticket_id": "T-1", "message": "changed"}),
            ),
        ):
            with self.subTest(expected_reason):
                engine = SecurityEngine()
                state = engine.propose(_action(), now=100, approval_ttl=20)
                result = engine.resume(
                    state,
                    decision,
                    current_action=current_action,
                )

                self.assertEqual((expected_status, expected_reason), (result.status, result.reason))
                self.assertEqual(0, engine.effects.mutation_count)

    def test_critical_action_requires_authorized_fresh_reviewer(self) -> None:
        engine = SecurityEngine()
        critical = _action(
            action_id="deploy-1",
            tool="deploy_production",
            resource="service:billing-api",
            arguments={"service": "billing-api", "version": "1.0.0"},
            environment="production",
        )
        state = engine.propose(critical, now=100)
        result = engine.resume(
            state,
            _decision(reviewer_role="developer", auth_age_seconds=10),
        )

        self.assertEqual(("denied", "reviewer_not_authorized"), (result.status, result.reason))
        self.assertEqual(0, engine.effects.mutation_count)

    def test_requester_cannot_approve_its_own_action(self) -> None:
        engine = SecurityEngine()
        state = engine.propose(_action(), now=100)
        result = engine.resume(state, _decision(reviewer_id="agent"))

        self.assertEqual(("denied", "self_approval_not_allowed"), (result.status, result.reason))
        self.assertEqual(0, engine.effects.mutation_count)

    def test_approval_time_and_auth_age_must_be_valid(self) -> None:
        for decision, reason in (
            (_decision(at=99), "invalid_approval_time"),
            (_decision(auth_age_seconds=-1), "invalid_auth_age"),
        ):
            with self.subTest(reason):
                engine = SecurityEngine()
                state = engine.propose(_action(), now=100)
                result = engine.resume(state, decision)

                self.assertEqual(("denied", reason), (result.status, result.reason))
                self.assertEqual(0, engine.effects.mutation_count)

    def test_reversible_action_requires_matching_rollback_contract(self) -> None:
        engine = SecurityEngine()
        action = _action(
            action_id="act-rollback",
            tool="update_ticket_label",
            arguments={"ticket_id": "T-1", "label": "needs-review"},
            rollback={
                "tool": "delete_ticket",
                "arguments": {"ticket_id": "T-2"},
            },
        )
        result = engine.propose(action, now=100)

        self.assertEqual(("denied", "invalid_rollback"), (result.status, result.reason))
        self.assertEqual(0, engine.effects.mutation_count)

    def test_exported_state_and_events_do_not_contain_body_or_credentials(self) -> None:
        engine = SecurityEngine(
            credentials={"send_customer_message": "secret-token-value"}
        )
        state = engine.propose(
            _action(arguments={"ticket_id": "T-1", "message": "private body"}),
            now=100,
        )
        state = engine.resume(state, _decision())
        exported = json.dumps(state.to_dict(), ensure_ascii=False)

        self.assertNotIn("private body", exported)
        self.assertNotIn("secret-token-value", exported)
        self.assertIn("arguments_hash", exported)


class SecurityEvaluationTests(unittest.TestCase):
    def test_all_cases_and_security_gates_pass(self) -> None:
        result = run_security_eval(CASES_PATH)

        self.assertTrue(result.gate_passed)
        self.assertEqual(8, result.total_cases)
        self.assertEqual(8, result.matched_cases)
        self.assertTrue(all(result.gate_checks.values()))

    def test_writer_creates_four_security_artifacts(self) -> None:
        result = run_security_eval(CASES_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_security_reports(result, Path(temporary))

            self.assertEqual(4, len(paths))
            self.assertTrue(all(path.exists() for path in paths))
            exported = "\n".join(path.read_text(encoding="utf-8") for path in paths)

        self.assertNotIn("secret-token-value", exported)
        self.assertNotIn("Original reviewed message", exported)

    def test_cli_policy_test_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_lab.py"),
                    "policy-test",
                    "--output",
                    temporary,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertTrue((Path(temporary) / "security-review.json").exists())
            self.assertIn('"gate_passed": true', completed.stdout)


def _action(
    *,
    action_id: str = "act-1",
    tool: str = "send_customer_message",
    resource: str = "ticket:T-1",
    arguments: dict | None = None,
    environment: str = "support",
    rollback: dict | None = None,
) -> ActionProposal:
    return ActionProposal(
        action_id=action_id,
        tool=tool,
        resource=resource,
        arguments=arguments
        or {"ticket_id": "T-1", "message": "reviewed message"},
        environment=environment,
        requester="agent",
        rollback=rollback,
    )


def _decision(
    *,
    decision_type: str = "approve",
    reviewer_role: str = "reviewer",
    at: int = 110,
    auth_age_seconds: int = 10,
    reviewer_id: str = "user-1",
) -> ApprovalDecision:
    return ApprovalDecision(
        decision=decision_type,
        reviewer_id=reviewer_id,
        reviewer_role=reviewer_role,
        decided_at=at,
        auth_age_seconds=auth_age_seconds,
    )


if __name__ == "__main__":
    unittest.main()
