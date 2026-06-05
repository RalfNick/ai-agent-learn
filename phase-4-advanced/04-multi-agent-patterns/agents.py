from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    DOC_RESEARCHER = "doc_researcher"
    CODE_ANALYST = "code_analyst"
    BENCHMARK_AGENT = "benchmark_agent"
    REVIEWER = "reviewer"


class ReviewStatus(str, Enum):
    APPROVED = "approved"
    NEEDS_EVIDENCE = "needs_evidence"
    NEEDS_REVISION = "needs_revision"


@dataclass
class SpecialistReport:
    role: AgentRole
    summary: str
    evidence: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


@dataclass
class ReviewResult:
    status: ReviewStatus
    score: float
    comments: list[str] = field(default_factory=list)


@dataclass
class MultiAgentResult:
    answer: str
    handoffs: list
    reports: list[SpecialistReport]
    evidence: list[str]
    review: ReviewResult
    trace: list[str]
