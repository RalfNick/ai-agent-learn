from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent_lab.memory import (
    MemoryCandidate,
    MemoryQuery,
    MemorySelector,
    MemoryStore,
    evaluate_candidate,
    load_memory_cases,
    run_memory_review,
)
from agent_lab.memory_reporting import write_memory_reports


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "datasets" / "memory-cases.jsonl"


class MemoryFixtureTests(unittest.TestCase):
    def test_dataset_covers_eight_beginner_cases(self) -> None:
        cases = load_memory_cases(CASES_PATH)

        self.assertEqual(8, len(cases))
        self.assertEqual(
            {
                "explicit-preference",
                "business-fact",
                "inferred-preference",
                "verified-workflow-lesson",
                "sensitive-value",
                "preference-correction",
                "cross-tenant-recall",
                "forget-preference",
            },
            {case.case_id for case in cases},
        )

    def test_candidate_policy_matches_the_first_five_cases(self) -> None:
        store = MemoryStore()
        cases = load_memory_cases(CASES_PATH)

        for case in cases[:5]:
            with self.subTest(case=case.case_id):
                self.assertIsNotNone(case.candidate)
                decision = evaluate_candidate(case.candidate, store)
                self.assertEqual(case.expected_action, decision.action)
                store.apply(decision)

        self.assertEqual(
            {"explicit-preference", "verified-workflow-lesson"},
            {record.created_by_case for record in store.active_records()},
        )

    def test_sensitive_candidate_never_reaches_a_record(self) -> None:
        case = _case("sensitive-value")
        decision = evaluate_candidate(case.candidate, MemoryStore())

        self.assertEqual("reject", decision.action)
        self.assertIsNone(decision.record)
        self.assertEqual("sensitive_or_prohibited", decision.reason)

    def test_duplicate_case_ids_are_rejected(self) -> None:
        lines = CASES_PATH.read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory() as temporary:
            duplicate_path = Path(temporary) / "duplicate.jsonl"
            duplicate_path.write_text(
                "\n".join([lines[0], lines[0]]) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate memory case id"):
                load_memory_cases(duplicate_path)


class MemoryLifecycleTests(unittest.TestCase):
    def test_new_explicit_preference_supersedes_the_old_version(self) -> None:
        store = MemoryStore()
        first = evaluate_candidate(_case("explicit-preference").candidate, store)
        store.apply(first)
        correction = evaluate_candidate(
            _case("preference-correction").candidate,
            store,
        )
        store.apply(correction)

        self.assertEqual("supersede", correction.action)
        records = store.records_for(
            ("tenant_acme", "user_u17"), "preference", "response_order"
        )
        self.assertEqual(["superseded", "active"], [item.status for item in records])
        self.assertEqual([1, 2], [item.version for item in records])

    def test_recall_filters_namespace_before_relevance(self) -> None:
        store = MemoryStore()
        store.apply(evaluate_candidate(_case("explicit-preference").candidate, store))

        allowed = store.recall(
            MemoryQuery(
                namespace=("tenant_acme", "user_u17"),
                kinds=("preference",),
                query_terms=("response", "evidence"),
                as_of="2026-08-04",
            )
        )
        blocked = store.recall(
            MemoryQuery(
                namespace=("tenant_beta", "user_u17"),
                kinds=("preference",),
                query_terms=("response", "evidence"),
                as_of="2026-08-04",
            )
        )

        self.assertEqual(1, len(allowed))
        self.assertEqual((), blocked)

    def test_expired_and_superseded_records_are_not_recalled(self) -> None:
        store = MemoryStore()
        expiring = MemoryCandidate(
            case_id="short-lived-lesson",
            namespace=("tenant_acme", "agent_support"),
            kind="procedure",
            key="temporary_route",
            statement="Use route v1 during the migration.",
            evidence_type="verified_reviewer",
            source_run_id="run_90",
            source_ref="review_8",
            reusable=True,
            sensitivity="internal",
            valid_until="2026-08-03",
            as_of="2026-08-01",
        )
        store.apply(evaluate_candidate(expiring, store))

        recalled = store.recall(
            MemoryQuery(
                namespace=("tenant_acme", "agent_support"),
                kinds=("procedure",),
                query_terms=("migration", "route"),
                as_of="2026-08-04",
            )
        )

        self.assertEqual((), recalled)

    def test_delete_purges_statement_and_keeps_content_free_tombstone(self) -> None:
        store = MemoryStore()
        store.apply(evaluate_candidate(_case("explicit-preference").candidate, store))
        selector = MemorySelector(
            case_id="forget-preference",
            namespace=("tenant_acme", "user_u17"),
            kind="preference",
            key="response_order",
            as_of="2026-08-04",
        )

        decision = store.delete(selector)
        records = store.records_for(
            selector.namespace, selector.kind, selector.key
        )

        self.assertEqual("delete", decision.action)
        self.assertTrue(records)
        self.assertTrue(all(record.status == "deleted" for record in records))
        self.assertTrue(all(record.statement is None for record in records))
        self.assertTrue(all(record.content_hash == "purged" for record in records))
        self.assertNotIn(
            "conclusion before the evidence",
            json.dumps([item.to_dict() for item in records]),
        )


class MemoryReviewTests(unittest.TestCase):
    def test_full_review_matches_all_declared_decisions(self) -> None:
        result = run_memory_review(CASES_PATH)

        self.assertTrue(result.gate_passed)
        self.assertEqual(8, result.total_cases)
        self.assertEqual(8, result.matched_cases)
        self.assertEqual(
            {"store": 2, "route_to_source": 1, "reject": 3, "supersede": 1, "delete": 1},
            result.decision_counts,
        )

    def test_writer_creates_five_redacted_artifacts(self) -> None:
        result = run_memory_review(CASES_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_memory_reports(result, Path(temporary))
            contents = [path.read_text(encoding="utf-8") for path in paths]

        self.assertEqual(
            [
                "memory-review.json",
                "memory-review.md",
                "memory-decisions.jsonl",
                "memory-store.jsonl",
                "memory-recall.md",
            ],
            [path.name for path in paths],
        )
        combined = "\n".join(contents)
        self.assertNotIn("reader@example.com", combined)
        self.assertNotIn("demo-secret-token", combined)
        self.assertNotIn("Put the conclusion before the evidence.", combined)

    def test_delete_gate_fails_when_selector_does_not_purge_any_record(self) -> None:
        cases = [
            json.loads(line)
            for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        delete_case = next(item for item in cases if item["id"] == "forget-preference")
        delete_case["key"] = "unknown_preference"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong-delete.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in cases) + "\n",
                encoding="utf-8",
            )
            result = run_memory_review(path)

        self.assertFalse(result.gate_passed)
        self.assertFalse(result.gate_checks["deleted_content_purged"])

    def test_memory_review_cli_exits_zero_and_prints_report_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    "run_lab.py",
                    "memory-review",
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
        self.assertIn('"gate_passed": true', completed.stdout)
        self.assertIn("memory-review.md", completed.stdout)


def _case(case_id: str):
    return next(
        case for case in load_memory_cases(CASES_PATH) if case.case_id == case_id
    )


if __name__ == "__main__":
    unittest.main()
