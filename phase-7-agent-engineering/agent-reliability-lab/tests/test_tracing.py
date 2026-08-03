from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent_lab.trace_reporting import write_trace_reports
from agent_lab.tracing import (
    TraceSpan,
    build_trace_for_case,
    load_trace_cases,
    review_trace,
    run_trace_review,
    sanitize_attributes,
)


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "datasets" / "trace-cases.jsonl"


class TraceFixtureTests(unittest.TestCase):
    def test_dataset_covers_eight_beginner_incidents(self) -> None:
        cases = load_trace_cases(CASES_PATH)

        self.assertEqual(8, len(cases))
        self.assertEqual(
            {
                "clean-run",
                "wrong-context",
                "safe-retry",
                "worker-resume",
                "missing-version",
                "orphan-span",
                "unclosed-span",
                "secret-leak",
            },
            {case.case_id for case in cases},
        )

    def test_sanitizer_keeps_only_allowed_low_risk_attributes(self) -> None:
        safe, findings = sanitize_attributes(
            {
                "source_ids": ["handbook:refund-policy"],
                "body_length": 182,
                "email": "reader@example.com",
            },
            allowed_keys={"source_ids", "body_length"},
        )

        self.assertEqual(
            {
                "source_ids": ["handbook:refund-policy"],
                "body_length": 182,
            },
            safe,
        )
        self.assertEqual(("disallowed_key:email",), findings)

    def test_sanitizer_detects_secret_like_value_even_on_allowed_key(self) -> None:
        safe, findings = sanitize_attributes(
            {"result_code": "Bearer demo-secret-token"},
            allowed_keys={"result_code"},
        )

        self.assertEqual({}, safe)
        self.assertEqual(("sensitive_value:result_code",), findings)


class TraceReviewTests(unittest.TestCase):
    def test_clean_trace_forms_one_readable_parent_child_tree(self) -> None:
        case = _case("clean-run")
        spans = build_trace_for_case(case)
        result = review_trace(case, spans)

        self.assertEqual(1, sum(span.parent_span_id is None for span in spans))
        self.assertEqual(
            {"agent", "retrieval", "model", "approval", "tool"},
            {span.kind for span in spans},
        )
        self.assertTrue(result.passed)
        self.assertFalse(result.findings)
        self.assertTrue(all(item.answered for item in result.questions))

    def test_review_detects_each_expected_fixture_problem(self) -> None:
        for case in load_trace_cases(CASES_PATH):
            with self.subTest(case=case.case_id):
                result = review_trace(case, build_trace_for_case(case))

                self.assertEqual(
                    set(case.expected_findings),
                    {finding.code for finding in result.findings},
                )
                self.assertTrue(result.passed)

    def test_safe_retry_contains_stable_action_and_receipt_evidence(self) -> None:
        case = _case("safe-retry")
        result = review_trace(case, build_trace_for_case(case))

        retry = next(span for span in result.spans if span.attempt == 2)
        self.assertIn("idempotency_key_hash", retry.attributes)
        self.assertEqual("committed", retry.attributes["receipt_status"])
        self.assertNotIn(
            "unsafe_retry", {finding.code for finding in result.findings}
        )

    def test_side_effecting_retry_without_evidence_is_rejected(self) -> None:
        case = _case("clean-run")
        spans = list(build_trace_for_case(case))
        spans.append(
            TraceSpan(
                trace_id=spans[0].trace_id,
                span_id="span-unsafe-retry",
                parent_span_id=spans[0].span_id,
                name="record_ticket_followup",
                kind="tool",
                sequence=99,
                duration_ms=5,
                status="ok",
                attempt=2,
                versions=spans[0].versions,
                attributes={"side_effecting": True},
                usage={"input_tokens": 0, "output_tokens": 0},
                error_code=None,
            )
        )

        result = review_trace(case, tuple(spans))

        self.assertIn("unsafe_retry", {item.code for item in result.findings})
        self.assertFalse(result.passed)

    def test_resume_keeps_trace_id_and_records_worker_handoff(self) -> None:
        case = _case("worker-resume")
        spans = build_trace_for_case(case)

        self.assertEqual(1, len({span.trace_id for span in spans}))
        self.assertEqual(len(spans), len({span.sequence for span in spans}))
        workers = {
            str(span.attributes["worker_id"])
            for span in spans
            if "worker_id" in span.attributes
        }
        self.assertEqual({"worker-a", "worker-b"}, workers)

    def test_sensitive_value_is_removed_from_reviewed_span(self) -> None:
        case = _case("secret-leak")
        result = review_trace(case, build_trace_for_case(case))

        tool = next(span for span in result.spans if span.kind == "tool")
        self.assertNotIn("result_code", tool.attributes)
        self.assertIn(
            "sensitive_attribute", {finding.code for finding in result.findings}
        )

    def test_full_review_passes_when_all_expected_defects_are_found(self) -> None:
        result = run_trace_review(CASES_PATH)

        self.assertTrue(result.gate_passed)
        self.assertEqual(8, result.total_cases)
        self.assertEqual(8, result.matched_cases)
        self.assertGreater(
            result.candidate_question_answer_rate,
            result.baseline_question_answer_rate,
        )


class TraceReportTests(unittest.TestCase):
    def test_writer_creates_four_review_artifacts_without_secret_value(self) -> None:
        result = run_trace_review(CASES_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            paths = write_trace_reports(result, output)
            contents = [path.read_text(encoding="utf-8") for path in paths]

        self.assertEqual(
            [
                "trace-review.json",
                "trace-review.md",
                "trace-failures.md",
                "traces.jsonl",
            ],
            [path.name for path in paths],
        )
        self.assertTrue(json.loads(contents[0])["gate_passed"])
        self.assertIn("Five debugging questions", contents[1])
        self.assertIn("wrong-context", contents[2])
        self.assertNotIn("demo-secret-token", "\n".join(contents))

    def test_trace_review_cli_exits_zero_and_prints_report_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    "run_lab.py",
                    "trace-review",
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
        self.assertIn("trace-review.md", completed.stdout)


def _case(case_id: str):
    return next(
        case for case in load_trace_cases(CASES_PATH) if case.case_id == case_id
    )


if __name__ == "__main__":
    unittest.main()
