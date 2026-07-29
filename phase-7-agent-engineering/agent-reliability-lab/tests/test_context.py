from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_lab.context import (
    assemble_context,
    estimate_tokens,
    load_context_cases,
    load_context_sources,
    run_context_eval,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "datasets" / "context-cases.jsonl"
SOURCES = ROOT / "fixtures" / "context" / "context-sources.jsonl"


class ContextTests(unittest.TestCase):
    def test_context_packet_improves_without_regressions(self) -> None:
        result = run_context_eval(CASES, SOURCES)
        self.assertTrue(result.gate_passed)
        self.assertGreater(len(result.improvements), 0)
        self.assertEqual(result.regressions, [])
        self.assertGreater(
            result.candidate.case_pass_rate,
            result.baseline.case_pass_rate,
        )

    def test_packet_filters_expired_untrusted_and_restricted_sources(self) -> None:
        result = run_context_eval(CASES, SOURCES)
        candidate_runs = {
            run.case.case_id: run
            for run in result.runs
            if run.strategy == "context-packet-v1"
        }
        reasons = {
            excluded.source_id: excluded.reason
            for run in candidate_runs.values()
            for excluded in run.packet.excluded
        }
        self.assertEqual(reasons["rate-limit-policy-expired"], "expired")
        self.assertEqual(reasons["external-pii-injection"], "untrusted")
        self.assertEqual(reasons["pii-policy-current"], "clearance")

    def test_missing_evidence_is_explicit(self) -> None:
        result = run_context_eval(CASES, SOURCES)
        run = next(
            run
            for run in result.runs
            if run.strategy == "context-packet-v1"
            and run.case.case_id == "ctx-005"
        )
        self.assertEqual(run.packet.selected, [])
        self.assertEqual(run.packet.missing_topics, ["expense-policy"])
        self.assertIn("<missing_evidence>", run.packet.rendered_context)

    def test_packet_respects_each_case_budget(self) -> None:
        result = run_context_eval(CASES, SOURCES)
        candidate_runs = [
            run for run in result.runs if run.strategy == "context-packet-v1"
        ]
        self.assertTrue(
            all(
                run.packet.budget_used <= run.packet.budget_limit
                for run in candidate_runs
            )
        )

    def test_too_small_budget_fails_the_release_gate(self) -> None:
        result = run_context_eval(CASES, SOURCES, budget_override=10)
        self.assertFalse(result.gate_passed)
        self.assertLess(result.candidate.required_topic_coverage, 1.0)

    def test_estimator_counts_chinese_and_code_deterministically(self) -> None:
        self.assertEqual(estimate_tokens("处理 PII: retry_count=3"), 6)
        self.assertEqual(
            estimate_tokens("处理 PII: retry_count=3"),
            estimate_tokens("处理 PII: retry_count=3"),
        )

    def test_duplicate_source_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.jsonl"
            row = (
                '{"id":"same","kind":"policy","title":"A","content":"B",'
                '"locator":"repo://a","trust":"trusted","authority":1,'
                '"updated_at":"2026-07-29","topics":["x"],"sensitivity":0}'
            )
            path.write_text(f"{row}\n{row}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate context source id"):
                load_context_sources(path)

    def test_missing_source_reference_is_rejected(self) -> None:
        cases = load_context_cases(CASES)
        sources = load_context_sources(SOURCES)
        source_map = {source.source_id: source for source in sources}
        broken = cases[0].__class__(
            **{
                **cases[0].__dict__,
                "candidate_source_ids": ["does-not-exist"],
            }
        )
        with self.assertRaisesRegex(ValueError, "references missing source"):
            assemble_context(
                broken,
                source_map,
                strategy="context-packet-v1",
            )


if __name__ == "__main__":
    unittest.main()
