from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    phase: str
    version: str


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=800)
    session_id: str = Field(default="default", min_length=1, max_length=80)

    @field_validator("question", "session_id")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class SourceItem(BaseModel):
    source_id: str
    title: str
    path: str
    score: float | None = None
    snippet: str | None = None


class TraceStep(BaseModel):
    step: str
    detail: str
    latency_ms: float | None = None


class AnswerResponse(BaseModel):
    question: str
    session_id: str
    answer: str
    mode: str
    sources: list[SourceItem]
    trace: list[TraceStep]
    review_status: str | None


class ObservabilitySummaryResponse(BaseModel):
    total_answer_requests: int
    last_session_id: str | None
    recent_questions: list[str]
