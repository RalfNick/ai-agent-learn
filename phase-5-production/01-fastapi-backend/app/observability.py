from __future__ import annotations

import logging
import re
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque

from app.schemas import AnswerResponse


LOGGER = logging.getLogger("phase5.agent_api")
TRACE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class HttpObservation:
    trace_id: str
    method: str
    path: str
    status_code: int
    latency_ms: float

    def to_public_dict(self) -> dict:
        data = asdict(self)
        data.pop("trace_id")
        return data


@dataclass(frozen=True)
class AgentRunObservation:
    trace_id: str
    question: str
    session_id: str
    latency_ms: float
    tool_count: int
    evidence_count: int
    review_status: str
    runtime_trace: list[str]
    estimated_cost_usd: float

    def to_public_dict(self) -> dict:
        data = asdict(self)
        data.pop("trace_id")
        return data


class ObservabilityStore:
    """In-memory observability store for Phase5 learning demos."""

    def __init__(self, max_items: int = 200) -> None:
        self.max_items = max_items
        self._requests: Deque[HttpObservation] = deque(maxlen=max_items)
        self._agent_runs: Deque[AgentRunObservation] = deque(maxlen=max_items)

    def record_http_request(self, observation: HttpObservation) -> None:
        self._requests.append(observation)
        LOGGER.info(
            "http_request trace_id=%s method=%s path=%s status_code=%s latency_ms=%.2f",
            observation.trace_id,
            observation.method,
            observation.path,
            observation.status_code,
            observation.latency_ms,
        )

    def record_agent_run(self, observation: AgentRunObservation) -> None:
        self._agent_runs.append(observation)
        LOGGER.info(
            "agent_run trace_id=%s session_id=%s review_status=%s tool_count=%s latency_ms=%.2f estimated_cost_usd=%.8f",
            observation.trace_id,
            observation.session_id,
            observation.review_status,
            observation.tool_count,
            observation.latency_ms,
            observation.estimated_cost_usd,
        )

    def summary(self) -> dict:
        request_latencies = [item.latency_ms for item in self._requests]
        agent_latencies = [item.latency_ms for item in self._agent_runs]
        recent_trace_ids = list(
            dict.fromkeys(
                [item.trace_id for item in reversed(self._requests)]
                + [item.trace_id for item in reversed(self._agent_runs)]
            )
        )[:10]

        return {
            "total_requests": len(self._requests),
            "total_agent_runs": len(self._agent_runs),
            "average_latency_ms": round(average(request_latencies), 2),
            "average_agent_latency_ms": round(average(agent_latencies), 2),
            "estimated_cost_usd": round(sum(item.estimated_cost_usd for item in self._agent_runs), 8),
            "recent_trace_ids": recent_trace_ids,
        }

    def trace_detail(self, trace_id: str) -> dict | None:
        normalized = normalize_trace_id(trace_id)
        request = next((item for item in reversed(self._requests) if item.trace_id == normalized), None)
        agent_run = next((item for item in reversed(self._agent_runs) if item.trace_id == normalized), None)

        if request is None and agent_run is None:
            return None

        return {
            "trace_id": normalized,
            "http": request.to_public_dict() if request else None,
            "agent": agent_run.to_public_dict() if agent_run else None,
        }


def normalize_trace_id(raw_trace_id: str | None) -> str:
    if raw_trace_id is None:
        return uuid.uuid4().hex
    normalized = TRACE_ID_PATTERN.sub("_", raw_trace_id.strip())[:80]
    return normalized or uuid.uuid4().hex


def now_ms() -> float:
    return time.perf_counter() * 1000


def elapsed_ms(start_ms: float) -> float:
    return round(now_ms() - start_ms, 2)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def estimate_answer_cost_usd(question: str, answer: AnswerResponse) -> float:
    """Rough deterministic estimate so Phase5 can observe cost shape before real LLM billing."""
    response_chars = len(answer.answer) + sum(len(item) for item in answer.evidence)
    request_chars = len(question)
    estimated_tokens = max(1, (request_chars + response_chars) / 4)
    return round(estimated_tokens / 1000 * 0.0002, 8)
