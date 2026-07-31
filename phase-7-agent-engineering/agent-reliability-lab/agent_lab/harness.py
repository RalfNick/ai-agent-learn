from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


EXPLICIT_STATES = {
    "completed",
    "failed",
    "failed_verification",
    "stopped",
    "waiting_approval",
}


@dataclass(frozen=True)
class ModelDecision:
    kind: str
    action_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    output: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelDecision:
        kind = str(data["kind"])
        if kind not in {"tool", "final"}:
            raise ValueError(f"unsupported model decision kind: {kind}")
        return cls(
            kind=kind,
            action_id=(
                str(data["action_id"]) if data.get("action_id") is not None else None
            ),
            tool_name=(
                str(data["tool"]) if data.get("tool") is not None else None
            ),
            arguments=dict(data.get("arguments", {})),
            output=str(data["output"]) if data.get("output") is not None else None,
        )


@dataclass(frozen=True)
class HarnessCase:
    case_id: str
    task: str
    script: list[ModelDecision]
    auto_approve: bool
    expected_status: str
    expected_side_effects: int
    required_events: list[str]
    ordered_events: list[tuple[str, str]]
    expected_failure_code: str | None


@dataclass
class RunState:
    run_id: str
    case_id: str
    strategy: str
    status: str = "ready"
    model_cursor: int = 0
    steps: int = 0
    pending_action: dict[str, Any] | None = None
    approvals: dict[str, bool] = field(default_factory=dict)
    completed_action_ids: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    final_output: str | None = None
    failure_code: str | None = None
    failure_detail: str | None = None
    checkpoint_count: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        return cls(
            run_id=str(data["run_id"]),
            case_id=str(data["case_id"]),
            strategy=str(data["strategy"]),
            status=str(data["status"]),
            model_cursor=int(data["model_cursor"]),
            steps=int(data["steps"]),
            pending_action=copy.deepcopy(data.get("pending_action")),
            approvals={
                str(key): bool(value)
                for key, value in dict(data.get("approvals", {})).items()
            },
            completed_action_ids=[
                str(item) for item in data.get("completed_action_ids", [])
            ],
            messages=copy.deepcopy(data.get("messages", [])),
            final_output=(
                str(data["final_output"])
                if data.get("final_output") is not None
                else None
            ),
            failure_code=(
                str(data["failure_code"])
                if data.get("failure_code") is not None
                else None
            ),
            failure_detail=(
                str(data["failure_detail"])
                if data.get("failure_detail") is not None
                else None
            ),
            checkpoint_count=int(data.get("checkpoint_count", 0)),
            events=copy.deepcopy(data.get("events", [])),
        )

    @classmethod
    def from_json(cls, raw: str) -> RunState:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class ToolResult:
    output: dict[str, Any]
    latency_ms: int
    side_effect: bool
    replayed: bool = False


@dataclass(frozen=True)
class HarnessGrade:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class HarnessRun:
    strategy: str
    case: HarnessCase
    state: RunState
    side_effect_count: int
    duplicate_side_effects: int
    grades: list[HarnessGrade]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "case": {
                "id": self.case.case_id,
                "expected_status": self.case.expected_status,
                "expected_side_effects": self.case.expected_side_effects,
                "auto_approve": self.case.auto_approve,
            },
            "state": self.state.to_dict(),
            "side_effect_count": self.side_effect_count,
            "duplicate_side_effects": self.duplicate_side_effects,
            "grades": [asdict(grade) for grade in self.grades],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class HarnessSummary:
    strategy: str
    cases: int
    passed_cases: int
    case_pass_rate: float
    explicit_state_rate: float
    sensitive_action_control_rate: float
    checkpoint_before_pause_rate: float
    trace_completeness_rate: float
    duplicate_side_effects: int


@dataclass(frozen=True)
class HarnessEvalResult:
    version: str
    baseline: HarnessSummary
    candidate: HarnessSummary
    improvements: list[str]
    regressions: list[str]
    gate_checks: dict[str, bool]
    gate_passed: bool
    runs: list[HarnessRun]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "comparison_scope": (
                "deterministic boundary conformance; not model quality or SDK ranking"
            ),
            "baseline": asdict(self.baseline),
            "candidate": asdict(self.candidate),
            "improvements": self.improvements,
            "regressions": self.regressions,
            "gate_checks": self.gate_checks,
            "gate_passed": self.gate_passed,
        }


class ToolTimeoutError(RuntimeError):
    pass


class ToolNotFoundError(RuntimeError):
    pass


class ScriptedModelAdapter:
    """A deterministic model seam used to test the runtime, not model quality."""

    adapter_id = "scripted-model-v1"

    def next_decision(
        self,
        case: HarnessCase,
        state: RunState,
    ) -> ModelDecision:
        if state.model_cursor >= len(case.script):
            raise RuntimeError("script_exhausted_without_final")
        decision = case.script[state.model_cursor]
        state.model_cursor += 1
        return decision


class EventRecorder:
    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        self.events = copy.deepcopy(events or [])

    def emit(self, event_type: str, **payload: Any) -> None:
        self.events.append(
            {
                "seq": len(self.events) + 1,
                "type": event_type,
                **payload,
            }
        )


class SessionStore:
    """In-memory persistence adapter; RunState remains JSON serializable."""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def save(self, state: RunState) -> None:
        self._states[state.run_id] = copy.deepcopy(state.to_dict())

    def load(self, run_id: str) -> RunState:
        try:
            return RunState.from_dict(self._states[run_id])
        except KeyError as exc:
            raise KeyError(f"unknown run id: {run_id}") from exc


class ToolExecutor:
    """Deterministic tools plus an external-style idempotency ledger."""

    def __init__(self) -> None:
        self.side_effects: list[dict[str, Any]] = []
        self.receipts: dict[str, ToolResult] = {}
        self.duplicate_attempts = 0

    def execute(
        self,
        action_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout_ms: int,
    ) -> ToolResult:
        if action_id in self.receipts:
            self.duplicate_attempts += 1
            previous = self.receipts[action_id]
            return ToolResult(
                output=copy.deepcopy(previous.output),
                latency_ms=previous.latency_ms,
                side_effect=previous.side_effect,
                replayed=True,
            )

        if tool_name == "lookup_policy":
            latency_ms = int(arguments.get("simulated_latency_ms", 20))
            result = ToolResult(
                output={
                    "answer": "Refund requests require an order id and reason.",
                    "source": "product-handbook.md#refunds",
                },
                latency_ms=latency_ms,
                side_effect=False,
            )
        elif tool_name == "slow_lookup":
            latency_ms = int(arguments.get("simulated_latency_ms", 1200))
            result = ToolResult(
                output={
                    "answer": "This result should not cross the timeout boundary.",
                    "source": "slow-fixture",
                },
                latency_ms=latency_ms,
                side_effect=False,
            )
        elif tool_name == "record_followup":
            latency_ms = int(arguments.get("simulated_latency_ms", 30))
            result = ToolResult(
                output={
                    "recorded": True,
                    "ticket_id": str(arguments.get("ticket_id", "unknown")),
                },
                latency_ms=latency_ms,
                side_effect=True,
            )
        else:
            raise ToolNotFoundError(tool_name)

        # This lab uses deterministic latency metadata rather than wall-clock sleeps.
        if result.latency_ms > timeout_ms:
            raise ToolTimeoutError(
                f"{tool_name} exceeded {timeout_ms} ms "
                f"(simulated={result.latency_ms} ms)"
            )

        if result.side_effect:
            self.side_effects.append(
                {
                    "action_id": action_id,
                    "tool": tool_name,
                    "arguments": copy.deepcopy(arguments),
                }
            )
        self.receipts[action_id] = result
        return result


class MinimalHarness:
    def __init__(
        self,
        *,
        max_steps: int,
        tool_timeout_ms: int,
        model: ScriptedModelAdapter | None = None,
        tools: ToolExecutor | None = None,
        sessions: SessionStore | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if tool_timeout_ms < 1:
            raise ValueError("tool_timeout_ms must be at least 1")
        self.max_steps = max_steps
        self.tool_timeout_ms = tool_timeout_ms
        self.model = model or ScriptedModelAdapter()
        self.tools = tools or ToolExecutor()
        self.sessions = sessions or SessionStore()

    def start(self, case: HarnessCase) -> RunState:
        state = RunState(
            run_id=f"harness::{case.case_id}",
            case_id=case.case_id,
            strategy="minimal-harness-v1",
            status="running",
            messages=[{"role": "user", "content": case.task}],
        )
        recorder = EventRecorder()
        recorder.emit("run_started", run_id=state.run_id)
        recorder.emit(
            "context_assembled",
            packet="fixture-context-v1",
            task=case.task,
        )
        return self._drive(case, state, recorder)

    def resume(
        self,
        case: HarnessCase,
        state: RunState,
        *,
        approve: bool,
    ) -> RunState:
        if state.status != "waiting_approval" or state.pending_action is None:
            raise ValueError("only a waiting approval run can be resumed")
        # Serialize and restore to prove the boundary does not depend on object identity.
        restored = RunState.from_json(state.to_json())
        recorder = EventRecorder(restored.events)
        action_id = str(restored.pending_action["action_id"])
        restored.approvals[action_id] = approve
        restored.status = "running"
        recorder.emit(
            "run_resumed",
            action_id=action_id,
            decision="approved" if approve else "rejected",
        )
        return self._drive(case, restored, recorder)

    def _drive(
        self,
        case: HarnessCase,
        state: RunState,
        recorder: EventRecorder,
    ) -> RunState:
        while True:
            if state.pending_action is not None:
                if not self._execute_pending(state, recorder):
                    return self._finalize_state(state, recorder)

            if state.steps >= self.max_steps:
                state.status = "stopped"
                state.failure_code = "max_steps"
                state.failure_detail = (
                    f"run reached configured max_steps={self.max_steps}"
                )
                recorder.emit(
                    "run_stopped",
                    reason="max_steps",
                    max_steps=self.max_steps,
                )
                self._checkpoint(state, recorder, "stopped")
                return self._finalize_state(state, recorder)

            recorder.emit(
                "model_called",
                adapter=self.model.adapter_id,
                cursor=state.model_cursor,
            )
            try:
                decision = self.model.next_decision(case, state)
            except RuntimeError as exc:
                state.status = "failed"
                state.failure_code = "model_protocol"
                state.failure_detail = str(exc)
                recorder.emit("model_failed", error=str(exc))
                self._checkpoint(state, recorder, "model_failed")
                return self._finalize_state(state, recorder)

            state.steps += 1
            recorder.emit(
                "model_decision",
                kind=decision.kind,
                step=state.steps,
            )
            if decision.kind == "final":
                state.final_output = decision.output or ""
                recorder.emit("final_received")
                verified = _verify_output(state.final_output)
                recorder.emit(
                    "verification_finished",
                    passed=verified,
                    verifier="evidence-marker-v1",
                )
                if verified:
                    state.status = "completed"
                else:
                    state.status = "failed_verification"
                    state.failure_code = "missing_evidence"
                    state.failure_detail = "final output must contain source="
                recorder.emit("run_finished", status=state.status)
                self._checkpoint(state, recorder, "final")
                return self._finalize_state(state, recorder)

            if not decision.action_id or not decision.tool_name:
                state.status = "failed"
                state.failure_code = "invalid_tool_request"
                state.failure_detail = "tool request requires action_id and tool"
                recorder.emit("model_failed", error=state.failure_detail)
                self._checkpoint(state, recorder, "invalid_tool_request")
                return self._finalize_state(state, recorder)

            side_effect = decision.tool_name == "record_followup"
            state.pending_action = {
                "action_id": decision.action_id,
                "tool": decision.tool_name,
                "arguments": decision.arguments,
                "side_effect": side_effect,
            }
            recorder.emit(
                "action_proposed",
                action_id=decision.action_id,
                tool=decision.tool_name,
                side_effect=side_effect,
            )
            needs_approval = side_effect
            decision_value = state.approvals.get(decision.action_id)
            recorder.emit(
                "policy_checked",
                action_id=decision.action_id,
                needs_approval=needs_approval,
                decision=(
                    "approved"
                    if decision_value is True
                    else "rejected"
                    if decision_value is False
                    else "pending"
                    if needs_approval
                    else "allowed"
                ),
            )
            if needs_approval and decision_value is None:
                state.status = "waiting_approval"
                self._checkpoint(state, recorder, "before_approval_pause")
                recorder.emit(
                    "approval_requested",
                    action_id=decision.action_id,
                    tool=decision.tool_name,
                )
                recorder.emit("run_paused", reason="approval")
                self._checkpoint(state, recorder, "paused_run")
                return self._finalize_state(state, recorder)

    def _execute_pending(
        self,
        state: RunState,
        recorder: EventRecorder,
    ) -> bool:
        action = state.pending_action
        if action is None:
            return True
        action_id = str(action["action_id"])
        tool_name = str(action["tool"])
        side_effect = bool(action["side_effect"])
        approval = state.approvals.get(action_id)
        if side_effect and approval is False:
            state.status = "failed"
            state.failure_code = "approval_rejected"
            state.failure_detail = f"{tool_name} was rejected"
            recorder.emit(
                "approval_rejected",
                action_id=action_id,
                tool=tool_name,
            )
            recorder.emit("run_finished", status=state.status)
            self._checkpoint(state, recorder, "approval_rejected")
            return False
        if side_effect and approval is not True:
            state.status = "waiting_approval"
            return False

        self._checkpoint(state, recorder, "before_tool")
        recorder.emit(
            "tool_started",
            action_id=action_id,
            tool=tool_name,
            timeout_ms=self.tool_timeout_ms,
        )
        try:
            result = self.tools.execute(
                action_id,
                tool_name,
                dict(action["arguments"]),
                timeout_ms=self.tool_timeout_ms,
            )
        except ToolTimeoutError as exc:
            state.status = "failed"
            state.failure_code = "tool_timeout"
            state.failure_detail = str(exc)
            recorder.emit(
                "tool_failed",
                action_id=action_id,
                tool=tool_name,
                error="tool_timeout",
            )
            recorder.emit("run_finished", status=state.status)
            self._checkpoint(state, recorder, "tool_timeout")
            return False
        except ToolNotFoundError as exc:
            state.status = "failed"
            state.failure_code = "tool_not_found"
            state.failure_detail = str(exc)
            recorder.emit(
                "tool_failed",
                action_id=action_id,
                tool=tool_name,
                error="tool_not_found",
            )
            recorder.emit("run_finished", status=state.status)
            self._checkpoint(state, recorder, "tool_not_found")
            return False

        state.messages.append(
            {
                "role": "tool",
                "action_id": action_id,
                "name": tool_name,
                "content": result.output,
            }
        )
        if action_id not in state.completed_action_ids:
            state.completed_action_ids.append(action_id)
        state.pending_action = None
        recorder.emit(
            "tool_finished",
            action_id=action_id,
            tool=tool_name,
            latency_ms=result.latency_ms,
            side_effect=result.side_effect,
            replayed=result.replayed,
        )
        self._checkpoint(state, recorder, "after_tool")
        return True

    def _checkpoint(
        self,
        state: RunState,
        recorder: EventRecorder,
        reason: str,
    ) -> None:
        state.checkpoint_count += 1
        recorder.emit(
            "checkpoint_saved",
            reason=reason,
            checkpoint=state.checkpoint_count,
        )
        state.events = copy.deepcopy(recorder.events)
        self.sessions.save(state)

    @staticmethod
    def _finalize_state(
        state: RunState,
        recorder: EventRecorder,
    ) -> RunState:
        state.events = copy.deepcopy(recorder.events)
        return state


def load_harness_cases(path: Path) -> list[HarnessCase]:
    cases = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        data = json.loads(raw_line)
        case_id = str(data["id"])
        if case_id in seen:
            raise ValueError(
                f"duplicate harness case on line {line_number}: {case_id}"
            )
        seen.add(case_id)
        expected = dict(data["expected"])
        status = str(expected["status"])
        if status not in EXPLICIT_STATES:
            raise ValueError(
                f"invalid expected status on line {line_number}: {status}"
            )
        cases.append(
            HarnessCase(
                case_id=case_id,
                task=str(data["task"]),
                script=[
                    ModelDecision.from_dict(item) for item in data["script"]
                ],
                auto_approve=bool(data.get("auto_approve", False)),
                expected_status=status,
                expected_side_effects=int(expected["side_effects"]),
                required_events=[
                    str(event) for event in expected.get("required_events", [])
                ],
                ordered_events=[
                    (str(pair[0]), str(pair[1]))
                    for pair in expected.get("ordered_events", [])
                ],
                expected_failure_code=(
                    str(expected["failure_code"])
                    if expected.get("failure_code") is not None
                    else None
                ),
            )
        )
    if not cases:
        raise ValueError("harness case dataset must not be empty")
    return cases


def run_harness_eval(
    cases_path: Path,
    *,
    max_steps: int = 3,
    tool_timeout_ms: int = 500,
) -> HarnessEvalResult:
    cases = load_harness_cases(cases_path)
    runs: list[HarnessRun] = []
    for case in cases:
        runs.append(
            _run_inline_case(
                case,
                safety_steps=max(max_steps * 4, len(case.script) + 1),
                tool_timeout_ms=tool_timeout_ms,
            )
        )
        runs.append(
            _run_harness_case(
                case,
                max_steps=max_steps,
                tool_timeout_ms=tool_timeout_ms,
            )
        )

    baseline_runs = [run for run in runs if run.strategy == "inline-loop-v1"]
    candidate_runs = [
        run for run in runs if run.strategy == "minimal-harness-v1"
    ]
    baseline = _summarize("inline-loop-v1", baseline_runs)
    candidate = _summarize("minimal-harness-v1", candidate_runs)
    baseline_pass = {run.case.case_id: run.passed for run in baseline_runs}
    candidate_pass = {run.case.case_id: run.passed for run in candidate_runs}
    improvements = sorted(
        case_id
        for case_id, passed in baseline_pass.items()
        if not passed and candidate_pass[case_id]
    )
    regressions = sorted(
        case_id
        for case_id, passed in baseline_pass.items()
        if passed and not candidate_pass[case_id]
    )
    candidate_by_id = {run.case.case_id: run for run in candidate_runs}
    gate_checks = {
        "all_candidate_cases_pass": all(run.passed for run in candidate_runs),
        "at_least_one_measured_improvement": bool(improvements),
        "no_case_regressions": not regressions,
        "approval_pauses_before_write": (
            candidate_by_id["approval-pause"].state.status
            == "waiting_approval"
            and candidate_by_id["approval-pause"].side_effect_count == 0
        ),
        "approved_resume_writes_once": (
            candidate_by_id["approval-resume"].state.status == "completed"
            and candidate_by_id["approval-resume"].side_effect_count == 1
            and candidate_by_id["approval-resume"].duplicate_side_effects == 0
        ),
        "timeout_is_explicit": (
            candidate_by_id["tool-timeout"].state.failure_code
            == "tool_timeout"
        ),
        "step_budget_is_enforced": (
            candidate_by_id["step-budget"].state.failure_code == "max_steps"
        ),
        "verification_failure_is_not_success": (
            candidate_by_id["verification-failure"].state.status
            == "failed_verification"
        ),
    }
    return HarnessEvalResult(
        version="0.4.0",
        baseline=baseline,
        candidate=candidate,
        improvements=improvements,
        regressions=regressions,
        gate_checks=gate_checks,
        gate_passed=all(gate_checks.values()),
        runs=runs,
    )


def _run_harness_case(
    case: HarnessCase,
    *,
    max_steps: int,
    tool_timeout_ms: int,
) -> HarnessRun:
    harness = MinimalHarness(
        max_steps=max_steps,
        tool_timeout_ms=tool_timeout_ms,
    )
    state = harness.start(case)
    if state.status == "waiting_approval" and case.auto_approve:
        state = harness.resume(case, state, approve=True)
    return _make_run(
        "minimal-harness-v1",
        case,
        state,
        len(harness.tools.side_effects),
        harness.tools.duplicate_attempts,
    )


def _run_inline_case(
    case: HarnessCase,
    *,
    safety_steps: int,
    tool_timeout_ms: int,
) -> HarnessRun:
    model = ScriptedModelAdapter()
    tools = ToolExecutor()
    state = RunState(
        run_id=f"inline::{case.case_id}",
        case_id=case.case_id,
        strategy="inline-loop-v1",
        status="running",
        messages=[{"role": "user", "content": case.task}],
    )
    recorder = EventRecorder()
    recorder.emit("run_started", run_id=state.run_id)
    recorder.emit("context_assembled", packet="raw-task-only")
    while state.steps < safety_steps:
        recorder.emit("model_called", adapter=model.adapter_id)
        try:
            decision = model.next_decision(case, state)
        except RuntimeError as exc:
            state.status = "failed"
            state.failure_code = "script_exhausted"
            state.failure_detail = str(exc)
            recorder.emit("run_finished", status=state.status)
            break
        state.steps += 1
        if decision.kind == "final":
            state.final_output = decision.output or ""
            state.status = "completed"
            recorder.emit("final_received")
            recorder.emit("run_finished", status=state.status)
            break
        action_id = decision.action_id or f"inline-{state.steps}"
        tool_name = decision.tool_name or "unknown"
        side_effect = tool_name == "record_followup"
        recorder.emit(
            "action_proposed",
            action_id=action_id,
            tool=tool_name,
            side_effect=side_effect,
        )
        recorder.emit(
            "tool_started",
            action_id=action_id,
            tool=tool_name,
            timeout_ms=None,
        )
        try:
            # The inline control intentionally omits timeout enforcement.
            result = tools.execute(
                action_id,
                tool_name,
                decision.arguments,
                timeout_ms=max(
                    tool_timeout_ms,
                    int(decision.arguments.get("simulated_latency_ms", 0)),
                ),
            )
        except (ToolNotFoundError, ToolTimeoutError) as exc:
            state.status = "failed"
            state.failure_code = "tool_error"
            state.failure_detail = str(exc)
            recorder.emit("tool_failed", action_id=action_id, tool=tool_name)
            recorder.emit("run_finished", status=state.status)
            break
        recorder.emit(
            "tool_finished",
            action_id=action_id,
            tool=tool_name,
            latency_ms=result.latency_ms,
            side_effect=result.side_effect,
        )
        state.messages.append(
            {
                "role": "tool",
                "action_id": action_id,
                "name": tool_name,
                "content": result.output,
            }
        )
    else:
        state.status = "failed"
        state.failure_code = "safety_stop"
        recorder.emit("run_finished", status=state.status)
    state.events = recorder.events
    return _make_run(
        "inline-loop-v1",
        case,
        state,
        len(tools.side_effects),
        tools.duplicate_attempts,
    )


def _make_run(
    strategy: str,
    case: HarnessCase,
    state: RunState,
    side_effect_count: int,
    duplicate_side_effects: int,
) -> HarnessRun:
    grades = _grade_run(case, state, side_effect_count)
    return HarnessRun(
        strategy=strategy,
        case=case,
        state=state,
        side_effect_count=side_effect_count,
        duplicate_side_effects=duplicate_side_effects,
        grades=grades,
        passed=all(grade.passed for grade in grades),
    )


def _grade_run(
    case: HarnessCase,
    state: RunState,
    side_effect_count: int,
) -> list[HarnessGrade]:
    event_types = [event["type"] for event in state.events]
    missing_events = sorted(set(case.required_events) - set(event_types))
    order_failures = [
        f"{before}->{after}"
        for before, after in case.ordered_events
        if not _event_before(event_types, before, after)
    ]
    grades = [
        HarnessGrade(
            name="expected_status",
            passed=state.status == case.expected_status,
            detail=f"actual={state.status}, expected={case.expected_status}",
        ),
        HarnessGrade(
            name="side_effect_count",
            passed=side_effect_count == case.expected_side_effects,
            detail=(
                f"actual={side_effect_count}, "
                f"expected={case.expected_side_effects}"
            ),
        ),
        HarnessGrade(
            name="required_events",
            passed=not missing_events,
            detail=(
                "all required events present"
                if not missing_events
                else f"missing={','.join(missing_events)}"
            ),
        ),
        HarnessGrade(
            name="event_order",
            passed=not order_failures,
            detail=(
                "all event ordering rules hold"
                if not order_failures
                else f"failed={','.join(order_failures)}"
            ),
        ),
    ]
    if case.expected_failure_code is not None:
        grades.append(
            HarnessGrade(
                name="failure_code",
                passed=state.failure_code == case.expected_failure_code,
                detail=(
                    f"actual={state.failure_code}, "
                    f"expected={case.expected_failure_code}"
                ),
            )
        )
    return grades


def _summarize(strategy: str, runs: list[HarnessRun]) -> HarnessSummary:
    explicit = sum(run.state.status in EXPLICIT_STATES for run in runs)
    sensitive_runs = [
        run
        for run in runs
        if any(
            event["type"] == "action_proposed" and event.get("side_effect")
            for event in run.state.events
        )
    ]
    controlled = sum(_sensitive_action_controlled(run) for run in sensitive_runs)
    approval_runs = [
        run
        for run in runs
        if "approval_requested"
        in [event["type"] for event in run.state.events]
        or run.case.case_id in {"approval-pause", "approval-resume"}
    ]
    checkpointed = sum(
        _event_before(
            [event["type"] for event in run.state.events],
            "checkpoint_saved",
            "approval_requested",
        )
        for run in approval_runs
    )
    completeness = []
    for run in runs:
        required = set(run.case.required_events)
        present = {event["type"] for event in run.state.events}
        completeness.append(
            len(required & present) / len(required) if required else 1.0
        )
    return HarnessSummary(
        strategy=strategy,
        cases=len(runs),
        passed_cases=sum(run.passed for run in runs),
        case_pass_rate=_rate(sum(run.passed for run in runs), len(runs)),
        explicit_state_rate=_rate(explicit, len(runs)),
        sensitive_action_control_rate=_rate(controlled, len(sensitive_runs)),
        checkpoint_before_pause_rate=_rate(checkpointed, len(approval_runs)),
        trace_completeness_rate=(
            round(sum(completeness) / len(completeness), 4)
            if completeness
            else 0.0
        ),
        duplicate_side_effects=sum(run.duplicate_side_effects for run in runs),
    )


def _sensitive_action_controlled(run: HarnessRun) -> bool:
    types = [event["type"] for event in run.state.events]
    policy_indexes = [
        index
        for index, event in enumerate(run.state.events)
        if event["type"] == "policy_checked"
    ]
    boundary_indexes = [
        index
        for index, event in enumerate(run.state.events)
        if event["type"] in {"approval_requested", "tool_started"}
    ]
    return bool(policy_indexes and boundary_indexes) and min(policy_indexes) < min(
        boundary_indexes
    )


def _event_before(events: list[str], before: str, after: str) -> bool:
    try:
        return events.index(before) < events.index(after)
    except ValueError:
        return False


def _verify_output(output: str) -> bool:
    return "source=" in output


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
