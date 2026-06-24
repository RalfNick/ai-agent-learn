import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.main import create_app


class AgentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_health_endpoint_returns_service_metadata(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["x-trace-id"])
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "phase5-agent-api")
        self.assertEqual(body["phase"], "phase-5")

    def test_answer_endpoint_wraps_phase4_runtime(self) -> None:
        response = self.client.post(
            "/api/v1/agent/answer",
            json={
                "question": "请结合 Phase4 Memory 的代码、文章和测试证据，说明是否可以进入 Phase5",
                "session_id": "test-session",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("Phase4 集成回答", body["answer"])
        self.assertEqual(body["review"]["status"], "approved")
        self.assertGreaterEqual(len(body["evidence"]), 3)
        self.assertIn("runtime.start", body["trace"])
        self.assertIn("reviewer.review", body["trace"])
        self.assertTrue(any(item["tool_name"] == "search_docs" for item in body["tool_results"]))
        self.assertTrue(any(item["tool_name"] == "find_code_examples" for item in body["tool_results"]))
        self.assertTrue(any(item["tool_name"] == "read_benchmark_summary" for item in body["tool_results"]))

    def test_observability_summary_records_http_and_agent_runs(self) -> None:
        response = self.client.post(
            "/api/v1/agent/answer",
            headers={"X-Trace-Id": "trace-summary-test"},
            json={
                "question": "请结合 Phase4 Memory 的代码和测试证据，说明当前状态",
                "session_id": "observability-summary",
            },
        )
        self.assertEqual(response.status_code, 200)

        summary_response = self.client.get("/api/v1/observability/summary")

        self.assertEqual(summary_response.status_code, 200)
        summary = summary_response.json()
        self.assertGreaterEqual(summary["total_requests"], 1)
        self.assertEqual(summary["total_agent_runs"], 1)
        self.assertGreater(summary["average_latency_ms"], 0)
        self.assertGreater(summary["average_agent_latency_ms"], 0)
        self.assertGreater(summary["estimated_cost_usd"], 0)
        self.assertIn("trace-summary-test", summary["recent_trace_ids"])

    def test_observability_trace_detail_contains_runtime_evidence(self) -> None:
        response = self.client.post(
            "/api/v1/agent/answer",
            headers={"X-Trace-Id": "trace-detail-test"},
            json={
                "question": "请结合 Phase4 Memory 的代码、文章和测试证据，说明是否可以进入 Phase5",
                "session_id": "observability-detail",
            },
        )
        self.assertEqual(response.status_code, 200)

        trace_response = self.client.get("/api/v1/observability/traces/trace-detail-test")

        self.assertEqual(trace_response.status_code, 200)
        trace = trace_response.json()
        self.assertEqual(trace["trace_id"], "trace-detail-test")
        self.assertEqual(trace["http"]["method"], "POST")
        self.assertEqual(trace["agent"]["review_status"], "approved")
        self.assertGreaterEqual(trace["agent"]["tool_count"], 3)
        self.assertGreaterEqual(trace["agent"]["evidence_count"], 3)
        self.assertIn("runtime.start", trace["agent"]["runtime_trace"])
        self.assertGreater(trace["agent"]["estimated_cost_usd"], 0)

    def test_evaluation_cases_endpoint_lists_builtin_regression_cases(self) -> None:
        response = self.client.get("/api/v1/evaluations/cases")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertGreaterEqual(len(body["cases"]), 3)
        first_case = body["cases"][0]
        self.assertIn("case_id", first_case)
        self.assertIn("question", first_case)
        self.assertIn("expected_review_status", first_case)
        self.assertIn("required_trace_steps", first_case)
        self.assertIn("minimum_evidence_count", first_case)

    def test_evaluation_run_replays_cases_and_returns_quality_summary(self) -> None:
        response = self.client.post(
            "/api/v1/evaluations/run",
            json={"case_ids": ["phase4-memory-evidence"], "session_prefix": "eval-test"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_cases"], 1)
        self.assertEqual(body["passed_cases"], 1)
        self.assertEqual(body["failed_cases"], 0)
        self.assertEqual(body["pass_rate"], 1.0)
        self.assertGreater(body["average_latency_ms"], 0)
        self.assertGreater(body["estimated_cost_usd"], 0)
        self.assertEqual(len(body["results"]), 1)

        result = body["results"][0]
        self.assertEqual(result["case_id"], "phase4-memory-evidence")
        self.assertTrue(result["passed"])
        self.assertEqual(result["review_status"], "approved")
        self.assertGreaterEqual(result["evidence_count"], result["minimum_evidence_count"])
        self.assertIn("runtime.start", result["runtime_trace"])
        self.assertTrue(result["trace_id"].startswith("eval-phase4-memory-evidence"))

        trace_response = self.client.get(f"/api/v1/observability/traces/{result['trace_id']}")
        self.assertEqual(trace_response.status_code, 200)
        trace = trace_response.json()
        self.assertIsNone(trace["http"])
        self.assertEqual(trace["agent"]["review_status"], "approved")
        self.assertGreaterEqual(trace["agent"]["tool_count"], 3)

    def test_evaluation_run_rejects_unknown_case_id(self) -> None:
        response = self.client.post(
            "/api/v1/evaluations/run",
            json={"case_ids": ["missing-case"], "session_prefix": "eval-test"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("unknown evaluation case id", response.json()["detail"])

    def test_answer_endpoint_rejects_empty_question(self) -> None:
        response = self.client.post(
            "/api/v1/agent/answer",
            json={"question": "   ", "session_id": "test-session"},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
