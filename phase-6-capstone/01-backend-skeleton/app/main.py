from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Protocol

from app.config import Settings, get_settings
from app.observability import ObservabilityStore
from app.runtime import AnswerRuntime
from app.schemas import (
    AnswerRequest,
    AnswerResponse,
    HealthResponse,
    ObservabilitySummaryResponse,
)


class AnswerRuntimeProtocol(Protocol):
    def answer(self, question: str, session_id: str) -> AnswerResponse:
        ...


def create_app(
    settings: Settings | None = None,
    runtime: AnswerRuntimeProtocol | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_runtime = runtime or AnswerRuntime()
    observability = ObservabilityStore()
    app = FastAPI(
        title="Phase6 Capstone API",
        version=resolved_settings.version,
        summary="Backend skeleton for the enterprise knowledge Agent capstone.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.state.observability = observability

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=resolved_settings.service_name,
            phase=resolved_settings.phase,
            version=resolved_settings.version,
        )

    @app.post("/api/v1/answer", response_model=AnswerResponse)
    def answer(request: AnswerRequest) -> AnswerResponse:
        response = resolved_runtime.answer(question=request.question, session_id=request.session_id)
        observability.record_answer_request(question=request.question, session_id=request.session_id)
        return response

    @app.get("/api/v1/observability/summary", response_model=ObservabilitySummaryResponse)
    def observability_summary() -> dict:
        return observability.summary()

    return app


app = create_app()
