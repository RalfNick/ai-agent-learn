from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
)

ALLOWED_ATTRIBUTE_KEYS = {
    "approval_state",
    "argument_hash",
    "body_length",
    "idempotency_key_hash",
    "prompt_hash",
    "receipt_status",
    "result_code",
    "resume_from_span",
    "route",
    "side_effecting",
    "source_ids",
    "worker_id",
}

REQUIRED_VERSION_KEYS = {"model", "prompt", "tool", "policy", "code"}
TERMINAL_SPAN_STATUSES = {"ok", "error", "cancelled"}
CRITICAL_VERSION_KINDS = {"agent", "model", "tool"}


@dataclass(frozen=True)
class TraceCase:
    case_id: str
    scenario: str
    description: str
    expected_source_ids: tuple[str, ...]
    expected_findings: tuple[str, ...]


@dataclass(frozen=True)
class TraceSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    sequence: int
    duration_ms: int
    status: str
    attempt: int
    versions: dict[str, str]
    attributes: dict[str, Any]
    usage: dict[str, int]
    error_code: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceFinding:
    code: str
    span_id: str | None
    detail: str


@dataclass(frozen=True)
class TraceQuestion:
    question: str
    answered: bool
    evidence: str


@dataclass(frozen=True)
class TraceCaseResult:
    case: TraceCase
    spans: tuple[TraceSpan, ...]
    findings: tuple[TraceFinding, ...]
    questions: tuple[TraceQuestion, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": {
                "id": self.case.case_id,
                "scenario": self.case.scenario,
                "description": self.case.description,
                "expected_findings": list(self.case.expected_findings),
            },
            "spans": [span.to_dict() for span in self.spans],
            "findings": [asdict(item) for item in self.findings],
            "questions": [asdict(item) for item in self.questions],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class TraceReviewResult:
    version: str
    total_cases: int
    matched_cases: int
    baseline_question_answer_rate: float
    candidate_question_answer_rate: float
    gate_checks: dict[str, bool]
    gate_passed: bool
    cases: tuple[TraceCaseResult, ...]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "comparison_scope": (
                "deterministic trace-contract fixtures; not a production "
                "observability benchmark"
            ),
            "total_cases": self.total_cases,
            "matched_cases": self.matched_cases,
            "baseline_question_answer_rate": self.baseline_question_answer_rate,
            "candidate_question_answer_rate": self.candidate_question_answer_rate,
            "gate_checks": self.gate_checks,
            "gate_passed": self.gate_passed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }


def load_trace_cases(path: Path) -> tuple[TraceCase, ...]:
    cases: list[TraceCase] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        data = json.loads(raw_line)
        try:
            cases.append(
                TraceCase(
                    case_id=str(data["id"]),
                    scenario=str(data["scenario"]),
                    description=str(data["description"]),
                    expected_source_ids=tuple(
                        str(item) for item in data["expected_source_ids"]
                    ),
                    expected_findings=tuple(
                        str(item) for item in data["expected_findings"]
                    ),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"trace case line {line_number} is missing {exc.args[0]}"
            ) from exc
    return tuple(cases)


def sanitize_attributes(
    attributes: dict[str, Any],
    *,
    allowed_keys: set[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    safe: dict[str, Any] = {}
    findings: list[str] = []
    for key, value in attributes.items():
        if key not in allowed_keys:
            findings.append(f"disallowed_key:{key}")
            continue
        if _contains_sensitive_value(value):
            findings.append(f"sensitive_value:{key}")
            continue
        safe[key] = value
    return safe, tuple(findings)


def build_trace_for_case(case: TraceCase) -> tuple[TraceSpan, ...]:
    trace_id = f"trace::{case.case_id}"
    versions = _versions()
    spans = [
        _span(
            trace_id,
            "root",
            None,
            "ticket_followup_agent",
            "agent",
            1,
            attributes={"worker_id": "worker-a"},
            versions=versions,
        ),
        _span(
            trace_id,
            "retrieve",
            "root",
            "retrieve_policy",
            "retrieval",
            2,
            attributes={"source_ids": ["handbook:refund-policy"]},
            versions=versions,
        ),
        _span(
            trace_id,
            "model",
            "root",
            "choose_followup_action",
            "model",
            3,
            attributes={
                "route": "record_ticket_followup",
                "prompt_hash": "sha256:prompt-v3",
            },
            usage={"input_tokens": 418, "output_tokens": 72},
            versions=versions,
        ),
        _span(
            trace_id,
            "approval",
            "root",
            "approve_external_write",
            "approval",
            4,
            attributes={"approval_state": "approved"},
            versions=versions,
        ),
        _span(
            trace_id,
            "tool",
            "root",
            "record_ticket_followup",
            "tool",
            5,
            attributes={
                "side_effecting": True,
                "argument_hash": "sha256:arguments-v1",
                "idempotency_key_hash": "sha256:action-t102",
                "receipt_status": "committed",
                "body_length": 182,
            },
            versions=versions,
        ),
    ]

    if case.scenario == "wrong_context":
        spans[0] = replace(spans[0], status="error", error_code="CHILD_FAILED")
        spans[1] = replace(
            spans[1], attributes={"source_ids": ["faq:refund-policy-2024"]}
        )
        spans[2] = replace(
            spans[2],
            attributes={
                "route": "close_ticket",
                "prompt_hash": "sha256:prompt-v3",
            },
        )
        spans[4] = replace(
            spans[4],
            name="close_ticket",
            status="error",
            error_code="WRONG_CONTEXT",
            attributes={
                "side_effecting": True,
                "argument_hash": "sha256:wrong-context",
                "idempotency_key_hash": "sha256:action-t102",
                "receipt_status": "rejected",
            },
        )
    elif case.scenario == "safe_retry":
        spans[4] = replace(
            spans[4],
            status="error",
            error_code="TOOL_TIMEOUT",
            attributes={
                "side_effecting": True,
                "argument_hash": "sha256:arguments-v1",
                "idempotency_key_hash": "sha256:action-t102",
                "receipt_status": "unknown",
            },
        )
        spans.extend(
            [
                _span(
                    trace_id,
                    "receipt",
                    "root",
                    "lookup_write_receipt",
                    "tool",
                    6,
                    attributes={
                        "side_effecting": False,
                        "receipt_status": "not_found",
                    },
                    versions=versions,
                ),
                _span(
                    trace_id,
                    "tool-retry",
                    "root",
                    "record_ticket_followup",
                    "tool",
                    7,
                    attempt=2,
                    attributes={
                        "side_effecting": True,
                        "argument_hash": "sha256:arguments-v1",
                        "idempotency_key_hash": "sha256:action-t102",
                        "receipt_status": "committed",
                    },
                    versions=versions,
                ),
            ]
        )
    elif case.scenario == "worker_resume":
        spans[3] = replace(
            spans[3],
            parent_span_id="resume",
            sequence=5,
        )
        spans.insert(
            4,
            _span(
                trace_id,
                "resume",
                "root",
                "resume_from_checkpoint",
                "agent",
                4,
                attributes={
                    "worker_id": "worker-b",
                    "resume_from_span": "model",
                },
                versions=versions,
            ),
        )
        spans[5] = replace(
            spans[5],
            parent_span_id="resume",
            sequence=6,
            attributes={
                **spans[5].attributes,
                "worker_id": "worker-b",
            },
        )
    elif case.scenario == "missing_version":
        spans[2] = replace(
            spans[2],
            versions={key: value for key, value in versions.items() if key != "prompt"},
        )
    elif case.scenario == "orphan_span":
        spans[4] = replace(spans[4], parent_span_id="missing-parent")
    elif case.scenario == "unclosed_span":
        spans[2] = replace(spans[2], status="unset")
    elif case.scenario == "secret_leak":
        spans[4] = replace(
            spans[4],
            attributes={
                **spans[4].attributes,
                "result_code": "Bearer demo-secret-token",
            },
        )
    return tuple(spans)


def review_trace(
    case: TraceCase, spans: tuple[TraceSpan, ...]
) -> TraceCaseResult:
    findings: list[TraceFinding] = []
    sanitized_spans: list[TraceSpan] = []
    span_by_id = {span.span_id: span for span in spans}

    roots = [span for span in spans if span.parent_span_id is None]
    if len(roots) != 1:
        findings.append(
            TraceFinding(
                "invalid_root_count",
                None,
                f"expected one root span, found {len(roots)}",
            )
        )

    for span in spans:
        safe_attributes, sanitation_findings = sanitize_attributes(
            span.attributes,
            allowed_keys=ALLOWED_ATTRIBUTE_KEYS,
        )
        if sanitation_findings:
            findings.append(
                TraceFinding(
                    "sensitive_attribute",
                    span.span_id,
                    ", ".join(sanitation_findings),
                )
            )
        sanitized = replace(span, attributes=safe_attributes)
        sanitized_spans.append(sanitized)

        if span.parent_span_id is not None:
            parent = span_by_id.get(span.parent_span_id)
            if parent is None:
                findings.append(
                    TraceFinding(
                        "orphan_span",
                        span.span_id,
                        f"parent {span.parent_span_id} does not exist",
                    )
                )
            elif parent.sequence >= span.sequence:
                findings.append(
                    TraceFinding(
                        "invalid_logical_order",
                        span.span_id,
                        "child sequence must be greater than parent sequence",
                    )
                )
        if span.status not in TERMINAL_SPAN_STATUSES:
            findings.append(
                TraceFinding(
                    "missing_terminal_status",
                    span.span_id,
                    f"status {span.status!r} is not terminal",
                )
            )
        if span.status == "error" and not span.error_code:
            findings.append(
                TraceFinding(
                    "missing_error_code",
                    span.span_id,
                    "error span must include a stable error code",
                )
            )
        if span.kind in CRITICAL_VERSION_KINDS:
            missing_versions = REQUIRED_VERSION_KEYS - set(span.versions)
            if missing_versions:
                findings.append(
                    TraceFinding(
                        "missing_version_evidence",
                        span.span_id,
                        "missing " + ", ".join(sorted(missing_versions)),
                    )
                )
        if any(value < 0 for value in span.usage.values()):
            findings.append(
                TraceFinding(
                    "invalid_usage",
                    span.span_id,
                    "usage values must not be negative",
                )
            )
        if (
            span.kind == "tool"
            and span.attempt > 1
            and bool(span.attributes.get("side_effecting"))
            and not (
                span.attributes.get("idempotency_key_hash")
                or span.attributes.get("receipt_status")
            )
        ):
            findings.append(
                TraceFinding(
                    "unsafe_retry",
                    span.span_id,
                    "side-effecting retry lacks idempotency or receipt evidence",
                )
            )

    retrieval_sources = {
        source
        for span in sanitized_spans
        if span.kind == "retrieval"
        for source in span.attributes.get("source_ids", [])
    }
    if set(case.expected_source_ids) - retrieval_sources:
        findings.append(
            TraceFinding(
                "context_mismatch",
                next(
                    (span.span_id for span in spans if span.kind == "retrieval"),
                    None,
                ),
                "selected sources do not satisfy the task contract",
            )
        )

    questions = _answer_questions(tuple(sanitized_spans))
    actual_codes = {finding.code for finding in findings}
    passed = actual_codes == set(case.expected_findings)
    return TraceCaseResult(
        case=case,
        spans=tuple(sanitized_spans),
        findings=tuple(findings),
        questions=questions,
        passed=passed,
    )


def run_trace_review(path: Path) -> TraceReviewResult:
    cases = load_trace_cases(path)
    results = tuple(
        review_trace(case, build_trace_for_case(case)) for case in cases
    )
    question_total = sum(len(case.questions) for case in results)
    answered = sum(
        question.answered for case in results for question in case.questions
    )
    candidate_rate = answered / question_total if question_total else 0.0
    safe_retry = next(case for case in results if case.case.case_id == "safe-retry")
    secret_case = next(case for case in results if case.case.case_id == "secret-leak")
    gate_checks = {
        "expected_findings_matched": all(case.passed for case in results),
        "clean_trace_has_no_findings": not next(
            case for case in results if case.case.case_id == "clean-run"
        ).findings,
        "sensitive_values_removed": all(
            not _contains_sensitive_value(span.attributes)
            for span in secret_case.spans
        ),
        "safe_retry_has_evidence": not any(
            finding.code == "unsafe_retry" for finding in safe_retry.findings
        ),
        "more_answerable_than_plain_log": candidate_rate > 0.2,
    }
    return TraceReviewResult(
        version="0.7.0",
        total_cases=len(results),
        matched_cases=sum(case.passed for case in results),
        baseline_question_answer_rate=0.2,
        candidate_question_answer_rate=candidate_rate,
        gate_checks=gate_checks,
        gate_passed=all(gate_checks.values()),
        cases=results,
    )


def _answer_questions(spans: tuple[TraceSpan, ...]) -> tuple[TraceQuestion, ...]:
    retrieval = [span for span in spans if span.kind == "retrieval"]
    tools = [span for span in spans if span.kind == "tool"]
    errors = [span for span in spans if span.status == "error"]
    retries = [span for span in tools if span.attempt > 1]
    critical = [span for span in spans if span.kind in CRITICAL_VERSION_KINDS]

    sources = sorted(
        {
            source
            for span in retrieval
            for source in span.attributes.get("source_ids", [])
        }
    )
    tool_path = " -> ".join(span.name for span in tools)
    first_error = min(errors, key=lambda span: span.sequence) if errors else None
    retries_safe = all(
        not span.attributes.get("side_effecting")
        or span.attributes.get("idempotency_key_hash")
        or span.attributes.get("receipt_status")
        for span in retries
    )
    versions_complete = all(
        REQUIRED_VERSION_KEYS <= set(span.versions) for span in critical
    )
    return (
        TraceQuestion(
            "context_sources",
            bool(sources),
            ", ".join(sources) if sources else "missing",
        ),
        TraceQuestion(
            "tool_path",
            bool(tools),
            tool_path if tool_path else "missing",
        ),
        TraceQuestion(
            "first_failure",
            not errors or bool(first_error and first_error.error_code),
            (
                f"{first_error.span_id}:{first_error.error_code}"
                if first_error
                else "no error span"
            ),
        ),
        TraceQuestion(
            "retry_evidence",
            retries_safe,
            "verified" if retries else "not applicable",
        ),
        TraceQuestion(
            "version_tuple",
            versions_complete,
            "complete" if versions_complete else "missing",
        ),
    )


def _span(
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    name: str,
    kind: str,
    sequence: int,
    *,
    duration_ms: int = 10,
    status: str = "ok",
    attempt: int = 1,
    versions: dict[str, str] | None = None,
    attributes: dict[str, Any] | None = None,
    usage: dict[str, int] | None = None,
    error_code: str | None = None,
) -> TraceSpan:
    return TraceSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name,
        kind=kind,
        sequence=sequence,
        duration_ms=duration_ms,
        status=status,
        attempt=attempt,
        versions=dict(versions or _versions()),
        attributes=dict(attributes or {}),
        usage=dict(usage or {"input_tokens": 0, "output_tokens": 0}),
        error_code=error_code,
    )


def _versions() -> dict[str, str]:
    return {
        "model": "fixture-model@1",
        "prompt": "followup@3",
        "tool": "ticket-tools@2",
        "policy": "approval@4",
        "code": "ae-07-trace",
    }


def _contains_sensitive_value(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return any(pattern.search(serialized) for pattern in SENSITIVE_VALUE_PATTERNS)
