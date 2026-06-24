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


class HttpObservationItem(BaseModel):
    method: str
    path: str
    status_code: int
    latency_ms: float


class AgentObservationItem(BaseModel):
    question: str
    session_id: str
    latency_ms: float
    tool_count: int
    evidence_count: int
    review_status: str
    runtime_trace: list[str]
    estimated_cost_usd: float


class ObservabilitySummaryResponse(BaseModel):
    total_requests: int
    total_agent_runs: int
    average_latency_ms: float
    average_agent_latency_ms: float
    estimated_cost_usd: float
    recent_trace_ids: list[str]


class TraceDetailResponse(BaseModel):
    trace_id: str
    http: HttpObservationItem | None
    agent: AgentObservationItem | None


class EvaluationRunRequest(BaseModel):
    case_ids: list[str] | None = None
    session_prefix: str = Field(default="eval", min_length=1, max_length=40)

    @field_validator("session_prefix")
    @classmethod
    def reject_blank_session_prefix(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class EvaluationCaseItem(BaseModel):
    case_id: str
    question: str
    expected_review_status: str
    required_trace_steps: list[str]
    minimum_evidence_count: int
    required_tool_names: list[str]


class EvaluationCasesResponse(BaseModel):
    cases: list[EvaluationCaseItem]


class EvaluationCaseResultItem(BaseModel):
    case_id: str
    trace_id: str
    passed: bool
    failures: list[str]
    latency_ms: float
    estimated_cost_usd: float
    review_status: str
    evidence_count: int
    minimum_evidence_count: int
    tool_names: list[str]
    runtime_trace: list[str]


class EvaluationRunResponse(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    average_latency_ms: float
    estimated_cost_usd: float
    results: list[EvaluationCaseResultItem]
