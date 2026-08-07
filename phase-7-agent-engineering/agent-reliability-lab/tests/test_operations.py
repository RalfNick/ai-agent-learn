from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_lab.operations import (
    OperationsPolicy,
    evaluate_window,
    incident_to_eval,
    load_operations_cases,
    run_operations_eval,
)
from agent_lab.operations_reporting import write_operations_reports


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "datasets" / "operations-cases.jsonl"


class OperationsFixtureTests(unittest.TestCase):
    def test_dataset_covers_eight_production_boundaries(self) -> None:
        cases = load_operations_cases(CASES_PATH)

        self.assertEqual(8, len(cases))
        self.assertEqual(
            {
                "stable-production",
                "tool-throttle-degrade",
                "latency-queue-degrade",
                "task-budget-stop",
                "unknown-write-pause",
                "provider-unavailable",
                "canary-regression",
                "canary-promote",
            },
            {case.case_id for case in cases},
        )

    def test_loader_rejects_duplicate_ids_unknown_fields_and_invalid_metrics(self) -> None:
        valid = json.loads(CASES_PATH.read_text(encoding="utf-8").splitlines()[0])
        variants = []
        variants.append([valid, valid])
        unknown = dict(valid)
        unknown["mystery"] = True
        variants.append([unknown])
        invalid = json.loads(json.dumps(valid))
        invalid["metrics"]["success_rate"] = 1.1
        variants.append([invalid])
        full_traffic_canary = json.loads(
            next(
                line
                for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
                if '"id":"canary-promote"' in line
            )
        )
        full_traffic_canary["traffic_percent"] = 100
        variants.append([full_traffic_canary])

        for rows in variants:
            with self.subTest(rows=rows):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "cases.jsonl"
                    path.write_text(
                        "\n".join(json.dumps(row) for row in rows) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        load_operations_cases(path)


class OperationsDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = OperationsPolicy()
        self.cases = {case.case_id: case for case in load_operations_cases(CASES_PATH)}

    def test_healthy_window_continues_without_incident(self) -> None:
        decision = evaluate_window(self.cases["stable-production"], self.policy)

        self.assertEqual(("continue", "within_policy"), (decision.action, decision.reason))
        self.assertIsNone(decision.incident)
        self.assertIsNone(incident_to_eval(decision))

    def test_tool_latency_cost_and_provider_failures_choose_declared_degradation(self) -> None:
        expectations = {
            "tool-throttle-degrade": ("read_only", "tool_error_budget_exceeded"),
            "latency-queue-degrade": ("draft_only", "latency_slo_missed"),
            "task-budget-stop": ("handoff", "task_cost_budget_exceeded"),
            "provider-unavailable": ("handoff", "model_provider_unavailable"),
        }

        for case_id, expected in expectations.items():
            with self.subTest(case_id=case_id):
                decision = evaluate_window(self.cases[case_id], self.policy)
                self.assertEqual(expected, (decision.action, decision.reason))
                self.assertIsNotNone(decision.incident)

    def test_unknown_write_has_priority_over_every_lower_risk_signal(self) -> None:
        decision = evaluate_window(self.cases["unknown-write-pause"], self.policy)

        self.assertEqual(("pause_writes", "write_outcome_unknown"), (decision.action, decision.reason))
        self.assertIn("write_outcome_unknown", decision.signals)
        self.assertIn("task_cost_budget_exceeded", decision.signals)

    def test_canary_online_regression_overrides_offline_eval_gain(self) -> None:
        decision = evaluate_window(self.cases["canary-regression"], self.policy)

        self.assertEqual(("rollback", "canary_error_regression"), (decision.action, decision.reason))
        self.assertGreater(
            self.cases["canary-regression"].release.candidate_eval_pass_rate,
            self.cases["canary-regression"].release.baseline_eval_pass_rate,
        )

    def test_healthy_canary_promotes_only_after_all_gates_pass(self) -> None:
        decision = evaluate_window(self.cases["canary-promote"], self.policy)

        self.assertEqual(("promote", "release_gates_passed"), (decision.action, decision.reason))
        self.assertEqual((), decision.signals)

    def test_canary_runtime_slo_failure_rolls_back_instead_of_degrading(self) -> None:
        healthy = self.cases["canary-promote"]
        slow_canary = replace(
            healthy,
            metrics=replace(healthy.metrics, p95_latency_ms=7000),
        )

        decision = evaluate_window(slow_canary, self.policy)

        self.assertEqual(("rollback", "latency_slo_missed"), (decision.action, decision.reason))

    def test_incident_eval_is_stable_and_redacted(self) -> None:
        decision = evaluate_window(self.cases["tool-throttle-degrade"], self.policy)
        first = incident_to_eval(decision)
        second = incident_to_eval(decision)

        self.assertEqual(first, second)
        exported = json.dumps(first.to_dict(), ensure_ascii=False, sort_keys=True)
        self.assertIn("source_hash", exported)
        self.assertNotIn("prompt", exported.lower())
        self.assertNotIn("customer", exported.lower())
        self.assertNotIn("token", exported.lower())


class OperationsEvaluationTests(unittest.TestCase):
    def test_all_cases_and_release_checks_pass(self) -> None:
        result = run_operations_eval(CASES_PATH)

        self.assertEqual((8, 8), (result.matched_cases, result.total_cases))
        self.assertTrue(result.gate_passed)
        self.assertTrue(all(result.gate_checks.values()))
        self.assertEqual(6, len(result.eval_candidates))

    def test_writer_creates_five_redacted_artifacts(self) -> None:
        result = run_operations_eval(CASES_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_operations_reports(result, Path(temporary))
            contents = "\n".join(path.read_text(encoding="utf-8") for path in paths)

        self.assertEqual(5, len(paths))
        self.assertEqual(
            {
                "operations-review.json",
                "operations-review.md",
                "operations-runs.jsonl",
                "incident-evals.jsonl",
                "operations-failures.md",
            },
            {path.name for path in paths},
        )
        self.assertNotIn("authorization", contents.lower())
        self.assertNotIn("api_key", contents.lower())

    def test_cli_ops_loop_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_lab.py"),
                    "ops-loop",
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


if __name__ == "__main__":
    unittest.main()
