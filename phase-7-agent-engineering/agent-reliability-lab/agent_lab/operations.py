from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OperationsPolicy:
    min_success_rate: float = 0.95
    max_p95_latency_ms: int = 5000
    max_cost_per_task: float = 0.30
    max_queue_age_seconds: int = 120
    max_tool_error_rate: float = 0.10
    max_handoff_rate: float = 0.40
    min_candidate_eval_pass_rate: float = 0.95
    max_canary_error_rate_delta: float = 0.03
    max_canary_cost_ratio: float = 1.20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowMetrics:
    success_rate: float
    p95_latency_ms: int
    cost_per_task: float
    queue_age_seconds: int
    tool_error_rate: float
    handoff_rate: float
    unknown_write_count: int
    provider_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseEvidence:
    baseline_eval_pass_rate: float
    candidate_eval_pass_rate: float
    safety_regressions: int
    control_error_rate: float
    canary_error_rate: float
    control_cost_per_task: float
    canary_cost_per_task: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperationsCase:
    case_id: str
    version: str
    stage: str
    traffic_percent: int
    metrics: WindowMetrics
    release: ReleaseEvidence | None
    expected_action: str
    expected_reason: str
    expected_eval_candidate: bool


@dataclass(frozen=True)
class Incident:
    incident_id: str
    case_id: str
    version: str
    action: str
    reason: str
    signals: tuple[str, ...]
    trace_ref_hash: str
    confirmed: bool = True
    redacted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalCandidate:
    task_id: str
    source_hash: str
    objective: str
    expected_control_action: str
    source_type: str = "confirmed_production_incident"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperationsDecision:
    case_id: str
    version: str
    stage: str
    action: str
    reason: str
    signals: tuple[str, ...]
    incident: Incident | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "version": self.version,
            "stage": self.stage,
            "action": self.action,
            "reason": self.reason,
            "signals": list(self.signals),
            "incident": self.incident.to_dict() if self.incident else None,
        }


@dataclass(frozen=True)
class OperationsCaseResult:
    case_id: str
    expected_action: str
    expected_reason: str
    expected_eval_candidate: bool
    decision: OperationsDecision
    eval_candidate: EvalCandidate | None
    matched: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "expected_action": self.expected_action,
            "expected_reason": self.expected_reason,
            "expected_eval_candidate": self.expected_eval_candidate,
            "decision": self.decision.to_dict(),
            "eval_candidate": (
                self.eval_candidate.to_dict() if self.eval_candidate else None
            ),
            "matched": self.matched,
        }


@dataclass(frozen=True)
class OperationsEvalResult:
    version: str
    policy: OperationsPolicy
    total_cases: int
    matched_cases: int
    action_counts: dict[str, int]
    gate_checks: dict[str, bool]
    gate_passed: bool
    cases: tuple[OperationsCaseResult, ...]
    eval_candidates: tuple[EvalCandidate, ...]

    def case_by_id(self, case_id: str) -> OperationsCaseResult:
        return next(case for case in self.cases if case.case_id == case_id)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "comparison_scope": (
                "deterministic operations-policy fixtures; not a production SRE, "
                "capacity, security, or model-quality audit"
            ),
            "total_cases": self.total_cases,
            "matched_cases": self.matched_cases,
            "action_counts": self.action_counts,
            "eval_candidates": len(self.eval_candidates),
            "gate_checks": self.gate_checks,
            "gate_passed": self.gate_passed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary_dict(),
            "policy": self.policy.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "incident_evals": [item.to_dict() for item in self.eval_candidates],
        }


TOP_LEVEL_KEYS = {
    "id",
    "version",
    "stage",
    "traffic_percent",
    "metrics",
    "release",
    "expected_action",
    "expected_reason",
    "expected_eval_candidate",
}
METRIC_KEYS = {
    "success_rate",
    "p95_latency_ms",
    "cost_per_task",
    "queue_age_seconds",
    "tool_error_rate",
    "handoff_rate",
    "unknown_write_count",
    "provider_available",
}
RELEASE_KEYS = {
    "baseline_eval_pass_rate",
    "candidate_eval_pass_rate",
    "safety_regressions",
    "control_error_rate",
    "canary_error_rate",
    "control_cost_per_task",
    "canary_cost_per_task",
}
ACTIONS = {
    "continue",
    "read_only",
    "draft_only",
    "handoff",
    "pause_writes",
    "rollback",
    "promote",
}


def load_operations_cases(path: Path) -> tuple[OperationsCase, ...]:
    cases: list[OperationsCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"line {line_number}: case must be an object")
        unknown = set(raw) - TOP_LEVEL_KEYS
        required = TOP_LEVEL_KEYS - {"release"}
        missing = required - set(raw)
        if unknown or missing:
            raise ValueError(
                f"line {line_number}: invalid case keys; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        case_id = _required_string(raw, "id", line_number)
        if case_id in seen:
            raise ValueError(f"line {line_number}: duplicate case id {case_id}")
        seen.add(case_id)
        stage = _required_string(raw, "stage", line_number)
        if stage not in {"stable", "canary"}:
            raise ValueError(f"line {line_number}: invalid stage {stage}")
        traffic_percent = _integer(raw.get("traffic_percent"), "traffic_percent")
        if not 1 <= traffic_percent <= 100:
            raise ValueError("traffic_percent must be between 1 and 100")
        if stage == "canary" and traffic_percent >= 100:
            raise ValueError("canary traffic_percent must be below 100")
        metrics = _parse_metrics(raw.get("metrics"), line_number)
        release = _parse_release(raw.get("release"), line_number)
        if stage == "canary" and release is None:
            raise ValueError(f"line {line_number}: canary requires release evidence")
        if stage == "stable" and release is not None:
            raise ValueError(f"line {line_number}: stable window cannot contain release")
        expected_action = _required_string(raw, "expected_action", line_number)
        if expected_action not in ACTIONS:
            raise ValueError(f"line {line_number}: invalid expected action")
        expected_eval = raw.get("expected_eval_candidate")
        if not isinstance(expected_eval, bool):
            raise ValueError("expected_eval_candidate must be boolean")
        cases.append(
            OperationsCase(
                case_id=case_id,
                version=_required_string(raw, "version", line_number),
                stage=stage,
                traffic_percent=traffic_percent,
                metrics=metrics,
                release=release,
                expected_action=expected_action,
                expected_reason=_required_string(raw, "expected_reason", line_number),
                expected_eval_candidate=expected_eval,
            )
        )
    return tuple(cases)


def evaluate_window(
    case: OperationsCase,
    policy: OperationsPolicy,
) -> OperationsDecision:
    metrics = case.metrics
    signals: list[str] = []
    if metrics.unknown_write_count > 0:
        signals.append("write_outcome_unknown")
    if case.release is not None:
        release = case.release
        if release.safety_regressions > 0:
            signals.append("canary_safety_regression")
        if (
            release.candidate_eval_pass_rate < policy.min_candidate_eval_pass_rate
            or release.candidate_eval_pass_rate < release.baseline_eval_pass_rate
        ):
            signals.append("offline_eval_regression")
        if (
            release.canary_error_rate - release.control_error_rate
            > policy.max_canary_error_rate_delta
        ):
            signals.append("canary_error_regression")
        if release.control_cost_per_task == 0:
            if release.canary_cost_per_task > 0:
                signals.append("canary_cost_regression")
        elif (
            release.canary_cost_per_task / release.control_cost_per_task
            > policy.max_canary_cost_ratio
        ):
            signals.append("canary_cost_regression")
    if not metrics.provider_available:
        signals.append("model_provider_unavailable")
    if metrics.cost_per_task > policy.max_cost_per_task:
        signals.append("task_cost_budget_exceeded")
    if metrics.p95_latency_ms > policy.max_p95_latency_ms:
        signals.append("latency_slo_missed")
    if metrics.queue_age_seconds > policy.max_queue_age_seconds:
        signals.append("queue_age_exceeded")
    if metrics.tool_error_rate > policy.max_tool_error_rate:
        signals.append("tool_error_budget_exceeded")
    if metrics.success_rate < policy.min_success_rate:
        signals.append("success_slo_missed")
    if metrics.handoff_rate > policy.max_handoff_rate:
        signals.append("handoff_capacity_exceeded")

    action, reason = _choose_action(tuple(signals), case.stage)
    incident = None
    if action not in {"continue", "promote"}:
        incident = _build_incident(case, action, reason, tuple(signals))
    return OperationsDecision(
        case_id=case.case_id,
        version=case.version,
        stage=case.stage,
        action=action,
        reason=reason,
        signals=tuple(signals),
        incident=incident,
    )


def incident_to_eval(decision: OperationsDecision) -> EvalCandidate | None:
    incident = decision.incident
    if incident is None or not incident.confirmed or not incident.redacted:
        return None
    source_hash = _hash_value(incident.to_dict())
    return EvalCandidate(
        task_id=f"regression-{source_hash[:12]}",
        source_hash=source_hash,
        objective=(
            f"Reproduce control signal '{incident.reason}' and require "
            f"the '{incident.action}' operational action."
        ),
        expected_control_action=incident.action,
    )


def run_operations_eval(
    cases_path: Path,
    policy: OperationsPolicy | None = None,
) -> OperationsEvalResult:
    active_policy = policy or OperationsPolicy()
    results: list[OperationsCaseResult] = []
    candidates: list[EvalCandidate] = []
    for case in load_operations_cases(cases_path):
        decision = evaluate_window(case, active_policy)
        candidate = incident_to_eval(decision)
        matched = (
            decision.action == case.expected_action
            and decision.reason == case.expected_reason
            and (candidate is not None) == case.expected_eval_candidate
        )
        result = OperationsCaseResult(
            case_id=case.case_id,
            expected_action=case.expected_action,
            expected_reason=case.expected_reason,
            expected_eval_candidate=case.expected_eval_candidate,
            decision=decision,
            eval_candidate=candidate,
            matched=matched,
        )
        results.append(result)
        if candidate is not None:
            candidates.append(candidate)

    by_id = {case.case_id: case for case in results}
    gate_checks = {
        "expected_decisions": all(case.matched for case in results),
        "stable_continues": by_id["stable-production"].decision.action == "continue",
        "tool_degrades_read_only": (
            by_id["tool-throttle-degrade"].decision.action == "read_only"
        ),
        "latency_degrades_draft": (
            by_id["latency-queue-degrade"].decision.action == "draft_only"
        ),
        "budget_and_provider_handoff": all(
            by_id[case_id].decision.action == "handoff"
            for case_id in ("task-budget-stop", "provider-unavailable")
        ),
        "unknown_write_pauses": (
            by_id["unknown-write-pause"].decision.action == "pause_writes"
        ),
        "canary_regression_rolls_back": (
            by_id["canary-regression"].decision.action == "rollback"
        ),
        "healthy_canary_promotes": (
            by_id["canary-promote"].decision.action == "promote"
        ),
        "incident_evals_redacted": all(
            case.decision.incident is None
            or (
                case.decision.incident.confirmed
                and case.decision.incident.redacted
                and case.eval_candidate is not None
            )
            for case in results
        ),
    }
    action_counts: dict[str, int] = {}
    for case in results:
        action = case.decision.action
        action_counts[action] = action_counts.get(action, 0) + 1
    return OperationsEvalResult(
        version="production-loop-v1",
        policy=active_policy,
        total_cases=len(results),
        matched_cases=sum(case.matched for case in results),
        action_counts=action_counts,
        gate_checks=gate_checks,
        gate_passed=all(gate_checks.values()),
        cases=tuple(results),
        eval_candidates=tuple(candidates),
    )


def _choose_action(signals: tuple[str, ...], stage: str) -> tuple[str, str]:
    priority = (
        ("write_outcome_unknown", "pause_writes"),
        ("canary_safety_regression", "rollback"),
        ("canary_error_regression", "rollback"),
        ("offline_eval_regression", "rollback"),
        ("canary_cost_regression", "rollback"),
        ("model_provider_unavailable", "handoff"),
        ("task_cost_budget_exceeded", "handoff"),
        ("latency_slo_missed", "draft_only"),
        ("queue_age_exceeded", "draft_only"),
        ("tool_error_budget_exceeded", "read_only"),
        ("success_slo_missed", "draft_only"),
        ("handoff_capacity_exceeded", "draft_only"),
    )
    for reason, action in priority:
        if reason in signals:
            if stage == "canary":
                return "rollback", reason
            return action, reason
    if stage == "canary":
        return "promote", "release_gates_passed"
    return "continue", "within_policy"


def _build_incident(
    case: OperationsCase,
    action: str,
    reason: str,
    signals: tuple[str, ...],
) -> Incident:
    identity = _hash_value(
        {
            "case_id": case.case_id,
            "version": case.version,
            "action": action,
            "reason": reason,
        }
    )
    return Incident(
        incident_id=f"incident-{identity[:12]}",
        case_id=case.case_id,
        version=case.version,
        action=action,
        reason=reason,
        signals=signals,
        trace_ref_hash=_hash_value(
            {"case_id": case.case_id, "version": case.version}
        ),
    )


def _parse_metrics(value: Any, line_number: int) -> WindowMetrics:
    if not isinstance(value, dict) or set(value) != METRIC_KEYS:
        raise ValueError(f"line {line_number}: metrics must contain declared keys")
    success_rate = _rate(value["success_rate"], "success_rate")
    tool_error_rate = _rate(value["tool_error_rate"], "tool_error_rate")
    handoff_rate = _rate(value["handoff_rate"], "handoff_rate")
    provider_available = value["provider_available"]
    if not isinstance(provider_available, bool):
        raise ValueError("provider_available must be boolean")
    return WindowMetrics(
        success_rate=success_rate,
        p95_latency_ms=_nonnegative_integer(
            value["p95_latency_ms"], "p95_latency_ms"
        ),
        cost_per_task=_nonnegative_number(value["cost_per_task"], "cost_per_task"),
        queue_age_seconds=_nonnegative_integer(
            value["queue_age_seconds"], "queue_age_seconds"
        ),
        tool_error_rate=tool_error_rate,
        handoff_rate=handoff_rate,
        unknown_write_count=_nonnegative_integer(
            value["unknown_write_count"], "unknown_write_count"
        ),
        provider_available=provider_available,
    )


def _parse_release(value: Any, line_number: int) -> ReleaseEvidence | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != RELEASE_KEYS:
        raise ValueError(f"line {line_number}: release must contain declared keys")
    return ReleaseEvidence(
        baseline_eval_pass_rate=_rate(
            value["baseline_eval_pass_rate"], "baseline_eval_pass_rate"
        ),
        candidate_eval_pass_rate=_rate(
            value["candidate_eval_pass_rate"], "candidate_eval_pass_rate"
        ),
        safety_regressions=_nonnegative_integer(
            value["safety_regressions"], "safety_regressions"
        ),
        control_error_rate=_rate(value["control_error_rate"], "control_error_rate"),
        canary_error_rate=_rate(value["canary_error_rate"], "canary_error_rate"),
        control_cost_per_task=_nonnegative_number(
            value["control_cost_per_task"], "control_cost_per_task"
        ),
        canary_cost_per_task=_nonnegative_number(
            value["canary_cost_per_task"], "canary_cost_per_task"
        ),
    )


def _required_string(raw: dict[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {key} must be a non-empty string")
    return value


def _rate(value: Any, name: str) -> float:
    number = _number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _nonnegative_number(value: Any, name: str) -> float:
    number = _number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    integer = _integer(value, name)
    if integer < 0:
        raise ValueError(f"{name} must be nonnegative")
    return integer


def _hash_value(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
