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

    def test_answer_endpoint_rejects_empty_question(self) -> None:
        response = self.client.post(
            "/api/v1/agent/answer",
            json={"question": "   ", "session_id": "test-session"},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
