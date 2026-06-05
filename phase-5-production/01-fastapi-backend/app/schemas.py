from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    phase: str
    version: str


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: str = Field(default="default", min_length=1, max_length=80)

    @field_validator("question", "session_id")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class MemoryContextItem(BaseModel):
    memory_type: str
    subject: str
    content: str


class ToolResultItem(BaseModel):
    tool_name: str
    query: str
    count: int
    evidence: list[str]
    summary: str


class ReviewResponse(BaseModel):
    status: str
    score: float
    comments: list[str]


class AnswerResponse(BaseModel):
    question: str
    session_id: str
    answer: str
    memory_context: list[MemoryContextItem]
    written_memory: MemoryContextItem | None
    tool_results: list[ToolResultItem]
    evidence: list[str]
    review: ReviewResponse
    trace: list[str]
