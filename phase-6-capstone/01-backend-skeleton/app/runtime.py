from __future__ import annotations

from app.schemas import AnswerResponse, TraceStep


class AnswerRuntime:
    """Placeholder runtime for the first Phase6 service boundary."""

    def answer(self, question: str, session_id: str) -> AnswerResponse:
        return AnswerResponse(
            question=question,
            session_id=session_id,
            mode="placeholder",
            answer=(
                "Phase6 backend skeleton is ready. Real knowledge ingestion, retrieval, "
                "and LangGraph Agentic QA will be added in the next capstone slices."
            ),
            sources=[],
            trace=[
                TraceStep(step="request.received", detail="Accepted answer request."),
                TraceStep(step="runtime.placeholder", detail="Returned skeleton response without retrieval."),
                TraceStep(step="response.placeholder", detail="Sources and review are intentionally empty."),
            ],
            review_status=None,
        )
