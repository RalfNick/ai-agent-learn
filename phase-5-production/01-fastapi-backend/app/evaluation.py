from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.observability import AgentRunObservation, elapsed_ms, estimate_answer_cost_usd, now_ms
from app.runtime_adapter import RuntimeAdapter
from app.schemas import AnswerResponse


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_review_status: str
    required_trace_steps: list[str]
    minimum_evidence_count: int
    required_tool_names: list[str]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "expected_review_status": self.expected_review_status,
            "required_trace_steps": self.required_trace_steps,
            "minimum_evidence_count": self.minimum_evidence_count,
            "required_tool_names": self.required_tool_names,
        }


@dataclass(frozen=True)
class EvaluationCaseResult:
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

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "trace_id": self.trace_id,
            "passed": self.passed,
            "failures": self.failures,
            "latency_ms": self.latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
            "review_status": self.review_status,
            "evidence_count": self.evidence_count,
            "minimum_evidence_count": self.minimum_evidence_count,
            "tool_names": self.tool_names,
            "runtime_trace": self.runtime_trace,
        }


DEFAULT_EVALUATION_CASES = [
    EvaluationCase(
        case_id="phase4-memory-evidence",
        question="请结合 Phase4 Memory 的代码、文章和测试证据，说明是否可以进入 Phase5",
        expected_review_status="approved",
        required_trace_steps=["runtime.start", "memory.search", "reviewer.review"],
        minimum_evidence_count=3,
        required_tool_names=["search_docs", "find_code_examples", "read_benchmark_summary"],
    ),
    EvaluationCase(
        case_id="phase4-code-architecture",
        question="请从代码架构角度说明 Phase4 runtime 集成了哪些能力",
        expected_review_status="approved",
        required_trace_steps=["runtime.start", "supervisor.plan", "reviewer.review"],
        minimum_evidence_count=3,
        required_tool_names=["find_code_examples"],
    ),
    EvaluationCase(
        case_id="phase4-observability-readiness",
        question="请结合测试和证据说明 Phase5 observability 当前是否可用",
        expected_review_status="approved",
        required_trace_steps=["runtime.start", "memory.search", "reviewer.review"],
        minimum_evidence_count=2,
        required_tool_names=["read_benchmark_summary"],
    ),
]


class EvaluationRunner:
    def __init__(
        self,
        adapter: RuntimeAdapter,
        cases: list[EvaluationCase] | None = None,
        on_agent_run: Callable[[AgentRunObservation], None] | None = None,
    ) -> None:
        self.adapter = adapter
        self.cases = cases or DEFAULT_EVALUATION_CASES
        self.on_agent_run = on_agent_run

    def list_cases(self) -> list[dict]:
        return [case.to_dict() for case in self.cases]

    def run(self, case_ids: list[str] | None = None, session_prefix: str = "eval") -> dict:
        selected_cases = self._select_cases(case_ids)
        results = [self._run_case(item, session_prefix=session_prefix) for item in selected_cases]
        passed_cases = sum(1 for item in results if item.passed)
        total_cases = len(results)
        total_cost = sum(item.estimated_cost_usd for item in results)
        average_latency_ms = sum(item.latency_ms for item in results) / total_cases if total_cases else 0.0

        return {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": total_cases - passed_cases,
            "pass_rate": round(passed_cases / total_cases, 4) if total_cases else 0.0,
            "average_latency_ms": round(average_latency_ms, 2),
            "estimated_cost_usd": round(total_cost, 8),
            "results": [item.to_dict() for item in results],
        }

    def _select_cases(self, case_ids: list[str] | None) -> list[EvaluationCase]:
        if not case_ids:
            return self.cases

        known = {item.case_id: item for item in self.cases}
        unknown = [case_id for case_id in case_ids if case_id not in known]
        if unknown:
            raise ValueError(f"unknown evaluation case id: {', '.join(unknown)}")
        return [known[case_id] for case_id in case_ids]

    def _run_case(self, case: EvaluationCase, session_prefix: str) -> EvaluationCaseResult:
        trace_id = f"eval-{case.case_id}"
        session_id = f"{session_prefix}-{case.case_id}"
        start_ms = now_ms()
        answer = self.adapter.answer(question=case.question, session_id=session_id)
        latency_ms = elapsed_ms(start_ms)
        failures = evaluate_answer(case, answer)
        estimated_cost_usd = estimate_answer_cost_usd(case.question, answer)

        if self.on_agent_run is not None:
            self.on_agent_run(
                AgentRunObservation(
                    trace_id=trace_id,
                    question=case.question,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    tool_count=len(answer.tool_results),
                    evidence_count=len(answer.evidence),
                    review_status=answer.review.status,
                    runtime_trace=answer.trace,
                    estimated_cost_usd=estimated_cost_usd,
                )
            )

        return EvaluationCaseResult(
            case_id=case.case_id,
            trace_id=trace_id,
            passed=not failures,
            failures=failures,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost_usd,
            review_status=answer.review.status,
            evidence_count=len(answer.evidence),
            minimum_evidence_count=case.minimum_evidence_count,
            tool_names=[item.tool_name for item in answer.tool_results],
            runtime_trace=answer.trace,
        )


def evaluate_answer(case: EvaluationCase, answer: AnswerResponse) -> list[str]:
    failures: list[str] = []
    tool_names = {item.tool_name for item in answer.tool_results}
    trace_steps = set(answer.trace)

    if answer.review.status != case.expected_review_status:
        failures.append(
            f"review_status expected {case.expected_review_status}, got {answer.review.status}"
        )

    if len(answer.evidence) < case.minimum_evidence_count:
        failures.append(
            f"evidence_count expected >= {case.minimum_evidence_count}, got {len(answer.evidence)}"
        )

    for required_tool in case.required_tool_names:
        if required_tool not in tool_names:
            failures.append(f"missing tool: {required_tool}")

    for required_step in case.required_trace_steps:
        if required_step not in trace_steps:
            failures.append(f"missing trace step: {required_step}")

    return failures
