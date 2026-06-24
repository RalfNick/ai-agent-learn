import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
CAPSTONE_ROOT = BACKEND_ROOT.parents[0]
INGESTION_ROOT = CAPSTONE_ROOT / "02-knowledge-ingestion"
RUNTIME_ROOT = CAPSTONE_ROOT / "03-agentic-qa-runtime"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(INGESTION_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from app.main import create_app
from app.schemas import AnswerResponse, SourceItem, TraceStep
from agentic_qa import build_runtime_from_sources


class Phase6BackendApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_health_returns_capstone_metadata(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "phase6-capstone-api")
        self.assertEqual(body["phase"], "phase-6")
        self.assertEqual(body["version"], "0.1.0")

    def test_answer_returns_placeholder_contract(self) -> None:
        response = self.client.post(
            "/api/v1/answer",
            json={"question": "Phase6 capstone 要做什么？", "session_id": "demo-session"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["question"], "Phase6 capstone 要做什么？")
        self.assertEqual(body["session_id"], "demo-session")
        self.assertEqual(body["mode"], "placeholder")
        self.assertIn("backend skeleton", body["answer"])
        self.assertEqual(body["sources"], [])
        self.assertGreaterEqual(len(body["trace"]), 3)
        self.assertEqual(body["trace"][0]["step"], "request.received")
        self.assertEqual(body["trace"][-1]["step"], "response.placeholder")
        self.assertIsNone(body["review_status"])

    def test_answer_rejects_blank_question(self) -> None:
        response = self.client.post(
            "/api/v1/answer",
            json={"question": "   ", "session_id": "demo-session"},
        )

        self.assertEqual(response.status_code, 422)

    def test_answer_endpoint_allows_local_web_ui_cors(self) -> None:
        origin = "http://127.0.0.1:3020"

        response = self.client.options(
            "/api/v1/answer",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        self.assertIn(response.status_code, {200, 204})
        self.assertEqual(origin, response.headers.get("access-control-allow-origin"))
        self.assertIn("POST", response.headers.get("access-control-allow-methods", ""))

    def test_observability_summary_counts_answer_requests(self) -> None:
        self.client.post(
            "/api/v1/answer",
            json={"question": "Phase6 当前做到哪一步？", "session_id": "summary-session"},
        )

        response = self.client.get("/api/v1/observability/summary")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_answer_requests"], 1)
        self.assertEqual(body["last_session_id"], "summary-session")
        self.assertEqual(body["recent_questions"], ["Phase6 当前做到哪一步？"])

    def test_answer_can_use_injected_agentic_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "agentic-rag.md").write_text(
                "\n".join(
                    [
                        "# Agentic Runtime",
                        "",
                        "Phase6 Agentic runtime 会返回 sources、trace 和 review_status。",
                        "它只基于检索证据回答。",
                    ]
                ),
                encoding="utf-8",
            )
            qa_runtime = build_runtime_from_sources([root], min_context_score=0.2, top_k=2)
            client = TestClient(create_app(runtime=AgenticRuntimeAdapter(qa_runtime)))

            response = client.post(
                "/api/v1/answer",
                json={"question": "Phase6 Agentic runtime 返回什么？", "session_id": "agentic-session"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual("agentic_rag", body["mode"])
        self.assertEqual("agentic-session", body["session_id"])
        self.assertEqual("evidence_supported", body["review_status"])
        self.assertGreaterEqual(len(body["sources"]), 1)
        self.assertEqual("Agentic Runtime", body["sources"][0]["title"])
        self.assertIn("retrieve", [step["step"] for step in body["trace"]])
        self.assertIn("answer.generate", [step["step"] for step in body["trace"]])


class AgenticRuntimeAdapter:
    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def answer(self, question: str, session_id: str) -> AnswerResponse:
        response = self.runtime.answer(question=question, session_id=session_id)
        return AnswerResponse(
            question=response.question,
            session_id=response.session_id,
            answer=response.answer,
            mode=response.mode,
            sources=[
                SourceItem(
                    source_id=source.source_id,
                    title=source.title,
                    path=source.path,
                    score=source.score,
                    snippet=source.snippet,
                )
                for source in response.sources
            ],
            trace=[
                TraceStep(
                    step=step.step,
                    detail=step.detail,
                    latency_ms=step.latency_ms,
                )
                for step in response.trace
            ],
            review_status=response.review_status,
        )


if __name__ == "__main__":
    unittest.main()
