from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import Settings, get_settings
from app.runtime_adapter import RuntimeAdapter
from app.schemas import AnswerRequest, AnswerResponse, HealthResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    adapter = RuntimeAdapter(resolved_settings)
    app = FastAPI(
        title="Phase5 Agent API",
        version=resolved_settings.version,
        summary="FastAPI wrapper around the Phase4 integrated Agent runtime.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=resolved_settings.service_name,
            phase=resolved_settings.phase,
            version=resolved_settings.version,
        )

    @app.post("/api/v1/agent/answer", response_model=AnswerResponse)
    def answer(request: AnswerRequest) -> AnswerResponse:
        try:
            return adapter.answer(question=request.question, session_id=request.session_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app()
