from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from agentic_qa import AgenticQARuntime
from agentic_qa.evidence import build_evidence_answer
from knowledge import build_index_from_paths


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    expected_terms: list[str]
    expected_review_status: str = "evidence_supported"
    expected_source_title: str | None = None
    forbidden_terms: list[str] = field(default_factory=list)
    expected_trace_steps: list[str] = field(default_factory=list)
    min_context_score: float | None = None
    top_k: int | None = None
    force_unsafe_answer: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCase":
        return cls(
            case_id=str(data["case_id"]),
            question=str(data["question"]),
            expected_terms=[str(term) for term in data.get("expected_terms", [])],
            expected_review_status=str(data.get("expected_review_status", "evidence_supported")),
            expected_source_title=(
                str(data["expected_source_title"])
                if data.get("expected_source_title") is not None
                else None
            ),
            forbidden_terms=[str(term) for term in data.get("forbidden_terms", [])],
            expected_trace_steps=[str(step) for step in data.get("expected_trace_steps", [])],
            min_context_score=(
                float(data["min_context_score"])
                if data.get("min_context_score") is not None
                else None
            ),
            top_k=int(data["top_k"]) if data.get("top_k") is not None else None,
            force_unsafe_answer=bool(data.get("force_unsafe_answer", False)),
        )


@dataclass(frozen=True)
class EvalRecord:
    case_id: str
    question: str
    passed: bool
    failures: list[str]
    review_status: str | None
    source_titles: list[str]
    trace_steps: list[str]
    answer: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvalSummary:
    total: int
    passed: int
    pass_rate: float
    records: list[EvalRecord]

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "records": [record.to_dict() for record in self.records],
        }


def evaluate_cases(
    cases: Sequence[EvalCase],
    source_paths: Sequence[Path | str],
    min_context_score: float = 0.2,
    top_k: int = 3,
) -> EvalSummary:
    index = build_index_from_paths(source_paths)
    records = []
    for case in cases:
        runtime = AgenticQARuntime(
            index=index,
            min_context_score=case.min_context_score
            if case.min_context_score is not None
            else min_context_score,
            top_k=case.top_k if case.top_k is not None else top_k,
            unsafe_answer_builder=unsafe_answer_builder if case.force_unsafe_answer else None,
        )
        response = runtime.answer(case.question, session_id=f"eval-{case.case_id}")
        failures = _case_failures(case, response)
        records.append(
            EvalRecord(
                case_id=case.case_id,
                question=case.question,
                passed=not failures,
                failures=failures,
                review_status=response.review_status,
                source_titles=[source.title for source in response.sources],
                trace_steps=[step.step for step in response.trace],
                answer=response.answer,
            )
        )
    passed = sum(1 for record in records if record.passed)
    pass_rate = round(passed / len(records), 4) if records else 0.0
    return EvalSummary(total=len(records), passed=passed, pass_rate=pass_rate, records=records)


def _case_failures(case: EvalCase, response) -> list[str]:
    failures: list[str] = []
    if response.review_status != case.expected_review_status:
        failures.append(
            f"review_status:{response.review_status}!={case.expected_review_status}"
        )
    normalized_answer = response.answer.lower()
    missing_terms = [
        term for term in case.expected_terms if term.lower() not in normalized_answer
    ]
    if missing_terms:
        failures.append(f"missing_terms:{','.join(missing_terms)}")
    forbidden_terms = [
        term for term in case.forbidden_terms if term.lower() in normalized_answer
    ]
    if forbidden_terms:
        failures.append(f"forbidden_terms:{','.join(forbidden_terms)}")
    if case.expected_source_title:
        source_titles = {source.title for source in response.sources}
        if case.expected_source_title not in source_titles:
            failures.append(f"missing_source:{case.expected_source_title}")
    missing_trace_steps = missing_ordered_steps(
        expected=case.expected_trace_steps,
        actual=[step.step for step in response.trace],
    )
    if missing_trace_steps:
        failures.append(f"missing_trace_steps:{','.join(missing_trace_steps)}")
    return failures


def missing_ordered_steps(expected: Sequence[str], actual: Sequence[str]) -> list[str]:
    missing: list[str] = []
    cursor = 0
    for expected_step in expected:
        try:
            next_index = actual.index(expected_step, cursor)
        except ValueError:
            missing.append(expected_step)
            continue
        cursor = next_index + 1
    return missing


def unsafe_answer_builder(question: str, results) -> str:
    return (
        build_evidence_answer(question, results)
        + "\n99. 公司报销制度要求发票抬头固定为测试公司。"
    )
