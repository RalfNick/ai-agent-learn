from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QASource:
    source_id: str
    title: str
    path: str
    score: float | None = None
    snippet: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QATraceStep:
    step: str
    detail: str
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QAResponse:
    question: str
    session_id: str
    answer: str
    mode: str
    sources: list[QASource]
    trace: list[QATraceStep]
    review_status: str | None
    context_score: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["sources"] = [source.to_dict() for source in self.sources]
        payload["trace"] = [step.to_dict() for step in self.trace]
        return payload
