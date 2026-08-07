from __future__ import annotations

import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

from agent_lab.graph import (
    GraphCase,
    GraphCompileError,
    GraphNode,
    compile_graph,
    load_graph_cases,
    run_graph_eval,
)


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "datasets" / "graph-cases.jsonl"


class GraphFixtureTests(unittest.TestCase):
    def test_dataset_covers_six_graph_boundaries(self) -> None:
        cases = load_graph_cases(CASES_PATH)

        self.assertEqual(6, len(cases))
        self.assertEqual(
            {
                "valid-diamond",
                "missing-dependency",
                "cycle-detected",
                "shared-write-conflict",
                "verifier-blocks-merge",
                "budget-exhausted",
            },
            {case.case_id for case in cases},
        )

    def test_valid_diamond_compiles_into_four_layers(self) -> None:
        case = _case("valid-diamond")

        compiled = compile_graph(case)

        self.assertEqual(
            (
                ("plan",),
                ("research-code", "research-docs", "research-policy"),
                ("verify",),
                ("merge",),
            ),
            compiled.layers,
        )

    def test_invalid_graphs_fail_for_the_declared_reason(self) -> None:
        for case_id, reason in (
            ("missing-dependency", "missing_dependency"),
            ("cycle-detected", "cycle_detected"),
            ("shared-write-conflict", "shared_write_conflict"),
        ):
            with self.subTest(case=case_id):
                with self.assertRaises(GraphCompileError) as raised:
                    compile_graph(_case(case_id))
                self.assertEqual(reason, raised.exception.reason)

    def test_verifier_requires_an_explicit_boolean_verdict(self) -> None:
        case = GraphCase(
            case_id="missing-verdict",
            expected_status="invalid",
            expected_reason="invalid_verifier",
            budget=2,
            nodes=(
                GraphNode("worker", "worker", (), {"finding": "ready"}, 1),
                GraphNode(
                    "verify",
                    "verifier",
                    ("worker",),
                    {"verification": "unknown"},
                    1,
                    verifies=("worker",),
                ),
            ),
        )

        with self.assertRaises(GraphCompileError) as raised:
            compile_graph(case)

        self.assertEqual("invalid_verifier", raised.exception.reason)

    def test_node_cost_must_be_positive(self) -> None:
        case = GraphCase(
            case_id="free-node",
            expected_status="invalid",
            expected_reason="invalid_node_cost",
            budget=1,
            nodes=(GraphNode("worker", "worker", (), {"finding": "ready"}, 0),),
        )

        with self.assertRaises(GraphCompileError) as raised:
            compile_graph(case)

        self.assertEqual("invalid_node_cost", raised.exception.reason)


class GraphExecutionTests(unittest.TestCase):
    def test_valid_diamond_merges_only_after_independent_verification(self) -> None:
        result = run_graph_eval(CASES_PATH)
        case = result.case_by_id("valid-diamond")

        self.assertEqual("completed", case.status)
        self.assertTrue(case.merge_executed)
        self.assertEqual("ready", case.final_state["article"])
        self.assertEqual("verified", case.final_state["verification"])
        self.assertLess(
            case.completed_nodes.index("verify"),
            case.completed_nodes.index("merge"),
        )

    def test_failed_verifier_blocks_merge(self) -> None:
        result = run_graph_eval(CASES_PATH)
        case = result.case_by_id("verifier-blocks-merge")

        self.assertEqual("blocked", case.status)
        self.assertEqual("verifier_failed", case.reason)
        self.assertFalse(case.merge_executed)
        self.assertNotIn("merge", case.completed_nodes)

    def test_budget_exhaustion_stops_before_merge(self) -> None:
        result = run_graph_eval(CASES_PATH)
        case = result.case_by_id("budget-exhausted")

        self.assertEqual("blocked", case.status)
        self.assertEqual("budget_exhausted", case.reason)
        self.assertLessEqual(case.spent_budget, case.budget)
        self.assertFalse(case.merge_executed)

    def test_release_gate_matches_all_cases_and_control_checks(self) -> None:
        result = run_graph_eval(CASES_PATH)

        self.assertTrue(result.gate_passed)
        self.assertEqual(6, result.total_cases)
        self.assertEqual(6, result.matched_cases)
        self.assertTrue(all(result.gate_checks.values()))


class GraphReportingTests(unittest.TestCase):
    def test_writer_creates_four_graph_artifacts(self) -> None:
        from agent_lab.graph_reporting import write_graph_reports

        result = run_graph_eval(CASES_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_graph_reports(result, Path(temporary))
            contents = [path.read_text(encoding="utf-8") for path in paths]

        self.assertEqual(
            [
                "graph-review.json",
                "graph-review.md",
                "graph-runs.jsonl",
                "graph-failures.md",
            ],
            [path.name for path in paths],
        )
        self.assertIn("valid-diamond", "\n".join(contents))

    def test_cli_graph_eval_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_lab.py"),
                    "graph-eval",
                    "--output",
                    temporary,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            report = Path(temporary) / "graph-review.json"

            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertTrue(report.exists())



def _case(case_id: str):
    return next(
        case for case in load_graph_cases(CASES_PATH) if case.case_id == case_id
    )


if __name__ == "__main__":
    unittest.main()
