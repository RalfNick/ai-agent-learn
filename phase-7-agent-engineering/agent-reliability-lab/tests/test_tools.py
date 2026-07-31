from __future__ import annotations

import unittest
from pathlib import Path

from agent_lab.tools import (
    ProposedCall,
    TicketStore,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    build_candidate_registry,
    load_tool_cases,
    run_tool_eval,
)


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "datasets" / "tool-cases.jsonl"


class ToolEngineeringTests(unittest.TestCase):
    def test_dataset_covers_nine_contract_cases(self) -> None:
        cases = load_tool_cases(CASES_PATH)

        self.assertEqual(9, len(cases))
        self.assertEqual(
            {
                "read-ticket",
                "preview-before-write",
                "write-needs-approval",
                "invalid-arguments",
                "permission-boundary",
                "duplicate-action",
                "idempotency-conflict",
                "transient-timeout",
                "bounded-list",
            },
            {case.case_id for case in cases},
        )

    def test_typed_registry_passes_release_gate(self) -> None:
        result = run_tool_eval(CASES_PATH)

        self.assertTrue(result.gate_passed)
        self.assertEqual(1.0, result.candidate.case_pass_rate)
        self.assertLess(result.baseline.case_pass_rate, 1.0)
        self.assertEqual(0, result.candidate.unsafe_side_effects)
        self.assertEqual(0, result.candidate.duplicate_side_effects)
        self.assertFalse(result.regressions)

    def test_schema_rejects_missing_note_before_handler(self) -> None:
        store = TicketStore()
        registry = build_candidate_registry(store)

        result = registry.invoke(
            ProposedCall(
                name="record_ticket_followup",
                arguments={"ticket_id": "T-102", "action_id": "write-102"},
            ),
            actor_permissions=frozenset({"ticket:write"}),
            approved=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual("invalid_arguments", result.error.code)
        self.assertEqual([], store.side_effects)

    def test_permission_and_approval_both_block_writes(self) -> None:
        call = ProposedCall(
            name="record_ticket_followup",
            arguments={
                "ticket_id": "T-102",
                "note": "Callback requested.",
                "action_id": "write-102",
            },
        )
        denied_store = TicketStore()
        denied = build_candidate_registry(denied_store).invoke(
            call,
            actor_permissions=frozenset({"ticket:read"}),
            approved=True,
        )
        paused_store = TicketStore()
        paused = build_candidate_registry(paused_store).invoke(
            call,
            actor_permissions=frozenset({"ticket:read", "ticket:write"}),
            approved=False,
        )

        self.assertEqual("permission_denied", denied.error.code)
        self.assertEqual("approval_required", paused.error.code)
        self.assertEqual([], denied_store.side_effects)
        self.assertEqual([], paused_store.side_effects)

    def test_idempotency_replays_receipt_without_second_write(self) -> None:
        store = TicketStore()
        registry = build_candidate_registry(store)
        call = ProposedCall(
            name="record_ticket_followup",
            arguments={
                "ticket_id": "T-102",
                "note": "Callback requested.",
                "action_id": "write-102",
            },
        )

        first = registry.invoke(
            call,
            actor_permissions=frozenset({"ticket:write"}),
            approved=True,
        )
        second = registry.invoke(
            call,
            actor_permissions=frozenset({"ticket:write"}),
            approved=True,
        )

        self.assertTrue(first.ok)
        self.assertTrue(second.replayed)
        self.assertEqual(first.output, second.output)
        self.assertEqual(1, len(store.side_effects))

    def test_idempotency_key_reuse_with_new_arguments_is_rejected(self) -> None:
        store = TicketStore()
        registry = build_candidate_registry(store)
        permissions = frozenset({"ticket:write"})
        first = ProposedCall(
            name="record_ticket_followup",
            arguments={
                "ticket_id": "T-102",
                "note": "First note.",
                "action_id": "write-102",
            },
        )
        conflicting = ProposedCall(
            name="record_ticket_followup",
            arguments={
                "ticket_id": "T-102",
                "note": "Different note.",
                "action_id": "write-102",
            },
        )

        registry.invoke(first, actor_permissions=permissions, approved=True)
        result = registry.invoke(
            conflicting,
            actor_permissions=permissions,
            approved=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual("idempotency_conflict", result.error.code)
        self.assertEqual(1, len(store.side_effects))

    def test_timeout_is_structured_and_retryable(self) -> None:
        registry = build_candidate_registry(TicketStore())

        result = registry.invoke(
            ProposedCall(
                name="slow_ticket_lookup",
                arguments={
                    "ticket_id": "T-102",
                    "simulated_latency_ms": 900,
                },
            ),
            actor_permissions=frozenset({"ticket:read"}),
            approved=False,
        )

        self.assertEqual("tool_timeout", result.error.code)
        self.assertTrue(result.error.retryable)
        self.assertEqual("dependency", result.error.category)

    def test_list_output_is_bounded_and_cursor_based(self) -> None:
        registry = build_candidate_registry(TicketStore())

        result = registry.invoke(
            ProposedCall(name="list_tickets", arguments={"limit": 3}),
            actor_permissions=frozenset({"ticket:read"}),
            approved=False,
        )

        self.assertTrue(result.ok)
        self.assertEqual(3, len(result.output["items"]))
        self.assertEqual("3", result.output["next_cursor"])

    def test_output_schema_is_a_runtime_boundary(self) -> None:
        store = TicketStore()
        registry = ToolRegistry(store)
        registry.register(
            ToolSpec(
                name="broken_tool",
                description="Return a deliberately invalid fixture.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
                required_permission="ticket:read",
            ),
            lambda _: ToolResult(ok=True, output={"unexpected": True}),
        )

        result = registry.invoke(
            ProposedCall(name="broken_tool", arguments={}),
            actor_permissions=frozenset({"ticket:read"}),
            approved=False,
        )

        self.assertFalse(result.ok)
        self.assertEqual("invalid_tool_output", result.error.code)

    def test_better_contract_has_an_explicit_context_cost(self) -> None:
        result = run_tool_eval(CASES_PATH)

        self.assertGreater(
            result.candidate.model_schema_bytes,
            result.baseline.model_schema_bytes,
        )


if __name__ == "__main__":
    unittest.main()
