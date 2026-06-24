from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from app.config import Settings, get_settings
from app.evaluation import EvaluationRunner
from app.observability import (
    AgentRunObservation,
    HttpObservation,
    ObservabilityStore,
    elapsed_ms,
    estimate_answer_cost_usd,
    normalize_trace_id,
    now_ms,
)
from app.runtime_adapter import RuntimeAdapter
from app.schemas import (
    AnswerRequest,
    AnswerResponse,
    EvaluationCasesResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    HealthResponse,
    ObservabilitySummaryResponse,
    TraceDetailResponse,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    adapter = RuntimeAdapter(resolved_settings)
    observability = ObservabilityStore()
    evaluator = EvaluationRunner(adapter, on_agent_run=observability.record_agent_run)
    app = FastAPI(
        title="Phase5 Agent API",
        version=resolved_settings.version,
        summary="FastAPI wrapper around the Phase4 integrated Agent runtime.",
    )
    app.state.observability = observability

    @app.middleware("http")
    async def add_trace_and_metrics(request: Request, call_next):
        trace_id = normalize_trace_id(request.headers.get("x-trace-id"))
        request.state.trace_id = trace_id
        start_ms = now_ms()

        try:
            response = await call_next(request)
        except Exception:
            observability.record_http_request(
                HttpObservation(
                    trace_id=trace_id,
                    method=request.method,
                    path=request.url.path,
                    status_code=500,
                    latency_ms=elapsed_ms(start_ms),
                )
            )
            raise

        response.headers["X-Trace-Id"] = trace_id
        observability.record_http_request(
            HttpObservation(
                trace_id=trace_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=elapsed_ms(start_ms),
            )
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=resolved_settings.service_name,
            phase=resolved_settings.phase,
            version=resolved_settings.version,
        )

    @app.post("/api/v1/agent/answer", response_model=AnswerResponse)
    def answer(payload: AnswerRequest, request: Request) -> AnswerResponse:
        start_ms = now_ms()
        try:
            response = adapter.answer(question=payload.question, session_id=payload.session_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        observability.record_agent_run(
            AgentRunObservation(
                trace_id=request.state.trace_id,
                question=payload.question,
                session_id=payload.session_id,
                latency_ms=elapsed_ms(start_ms),
                tool_count=len(response.tool_results),
                evidence_count=len(response.evidence),
                review_status=response.review.status,
                runtime_trace=response.trace,
                estimated_cost_usd=estimate_answer_cost_usd(payload.question, response),
            )
        )
        return response

    @app.get("/api/v1/observability/summary", response_model=ObservabilitySummaryResponse)
    def observability_summary() -> dict:
        return observability.summary()

    @app.get("/api/v1/observability/traces/{trace_id}", response_model=TraceDetailResponse)
    def observability_trace(trace_id: str) -> dict:
        detail = observability.trace_detail(trace_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"trace not found: {trace_id}")
        return detail

    @app.get("/api/v1/evaluations/cases", response_model=EvaluationCasesResponse)
    def evaluation_cases() -> dict:
        return {"cases": evaluator.list_cases()}

    @app.post("/api/v1/evaluations/run", response_model=EvaluationRunResponse)
    def evaluation_run(payload: EvaluationRunRequest) -> dict:
        try:
            return evaluator.run(case_ids=payload.case_ids, session_prefix=payload.session_prefix)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app()
