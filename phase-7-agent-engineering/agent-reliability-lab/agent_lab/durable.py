from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TERMINAL_STATES = {
    "cancelled",
    "completed",
    "failed",
    "waiting_reconciliation",
}


class ProcessCrash(RuntimeError):
    """A deterministic process-loss boundary used by the lab."""


class TransientStepError(RuntimeError):
    pass


class PermanentStepError(RuntimeError):
    pass


class ResultUnknownError(RuntimeError):
    pass


class StaleWorkerError(RuntimeError):
    pass


class IdempotencyConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class DurableCase:
    case_id: str
    fault: str
    expected_status: str
    expected_side_effects: int
    expected_duplicate_side_effects: int
    expected_model_attempts: int
    expected_failure_code: str | None
    required_events: tuple[str, ...]


@dataclass
class DurableRunState:
    run_id: str
    case_id: str
    strategy: str
    status: str = "ready"
    current_step: int = 0
    completed_steps: list[str] = field(default_factory=list)
    step_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    pending_action: dict[str, Any] | None = None
    cancel_requested: bool = False
    failure_code: str | None = None
    failure_detail: str | None = None
    next_retry_ms: int | None = None
    logical_clock_ms: int = 0
    lease_owner: str | None = None
    lease_epoch: int = 0
    checkpoint_count: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableRunState:
        return cls(
            run_id=str(data["run_id"]),
            case_id=str(data["case_id"]),
            strategy=str(data["strategy"]),
            status=str(data["status"]),
            current_step=int(data.get("current_step", 0)),
            completed_steps=[str(item) for item in data.get("completed_steps", [])],
            step_outputs=copy.deepcopy(dict(data.get("step_outputs", {}))),
            attempts={
                str(key): int(value)
                for key, value in dict(data.get("attempts", {})).items()
            },
            pending_action=copy.deepcopy(data.get("pending_action")),
            cancel_requested=bool(data.get("cancel_requested", False)),
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
            next_retry_ms=(
                int(data["next_retry_ms"])
                if data.get("next_retry_ms") is not None
                else None
            ),
            logical_clock_ms=int(data.get("logical_clock_ms", 0)),
            lease_owner=(
                str(data["lease_owner"])
                if data.get("lease_owner") is not None
                else None
            ),
            lease_epoch=int(data.get("lease_epoch", 0)),
            checkpoint_count=int(data.get("checkpoint_count", 0)),
            events=copy.deepcopy(data.get("events", [])),
        )


@dataclass(frozen=True)
class DurableGrade:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DurableRun:
    strategy: str
    case: DurableCase
    state: DurableRunState
    side_effect_count: int
    duplicate_side_effects: int
    blind_retries: int
    grades: tuple[DurableGrade, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "case": {
                "id": self.case.case_id,
                "fault": self.case.fault,
            },
            "state": self.state.to_dict(),
            "side_effect_count": self.side_effect_count,
            "duplicate_side_effects": self.duplicate_side_effects,
            "blind_retries": self.blind_retries,
            "grades": [asdict(grade) for grade in self.grades],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class DurableSummary:
    strategy: str
    cases: int
    passed_cases: int
    case_pass_rate: float
    total_model_attempts: int
    duplicate_side_effects: int
    blind_retries: int
    explicit_terminal_rate: float


@dataclass(frozen=True)
class DurableEvalResult:
    version: str
    baseline: DurableSummary
    candidate: DurableSummary
    improvements: tuple[str, ...]
    regressions: tuple[str, ...]
    gate_checks: dict[str, bool]
    gate_passed: bool
    runs: tuple[DurableRun, ...]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "comparison_scope": (
                "deterministic fault-injection contract test; not a distributed "
                "runtime benchmark"
            ),
            "baseline": asdict(self.baseline),
            "candidate": asdict(self.candidate),
            "improvements": list(self.improvements),
            "regressions": list(self.regressions),
            "gate_checks": self.gate_checks,
            "gate_passed": self.gate_passed,
        }


class JsonRunStore:
    """A tiny file-backed store with atomic replacement at the file boundary."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state: DurableRunState) -> None:
        target = self.root / f"{_safe_name(state.run_id)}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def load(self, run_id: str) -> DurableRunState:
        target = self.root / f"{_safe_name(run_id)}.json"
        return DurableRunState.from_dict(
            json.loads(target.read_text(encoding="utf-8"))
        )


class FaultPlan:
    def __init__(self, fault: str) -> None:
        self.fault = fault
        self.counts: dict[str, int] = {}

    def take(self, point: str, *, limit: int = 1) -> bool:
        if self.fault != point:
            return False
        count = self.counts.get(point, 0)
        if count >= limit:
            return False
        self.counts[point] = count + 1
        return True


class ScriptedModelService:
    """External-style deterministic model seam that survives runner rebuilds."""

    def __init__(self, faults: FaultPlan) -> None:
        self.faults = faults
        self.calls = 0

    def decide(self, case: DurableCase) -> dict[str, Any]:
        self.calls += 1
        if case.fault == "model_transient_twice" and self.calls <= 2:
            raise TransientStepError("simulated model gateway timeout")
        if case.fault == "model_transient_forever":
            raise TransientStepError("simulated persistent model gateway timeout")
        if case.fault == "model_permanent":
            raise PermanentStepError("simulated invalid model request")
        return {
            "tool": "record_ticket_followup",
            "arguments": {
                "ticket_id": "T-102",
                "note": f"Follow up generated for {case.case_id}.",
            },
        }


class TicketEffectStore:
    """External side effects and receipts are intentionally outside RunState."""

    def __init__(self, faults: FaultPlan) -> None:
        self.faults = faults
        self.effects: list[dict[str, Any]] = []
        self.receipts: dict[str, dict[str, Any]] = {}
        self.highest_fence = 0

    def activate_fence(self, epoch: int) -> None:
        self.highest_fence = max(self.highest_fence, epoch)

    def record_followup(
        self,
        *,
        action_id: str,
        logical_operation: str,
        payload: dict[str, Any],
        fence: int,
    ) -> dict[str, Any]:
        if fence < self.highest_fence:
            raise StaleWorkerError(
                f"fence={fence} is older than active fence={self.highest_fence}"
            )
        request_fingerprint = _fingerprint(payload)
        if action_id in self.receipts:
            previous = self.receipts[action_id]
            if previous["request_fingerprint"] != request_fingerprint:
                raise IdempotencyConflictError(
                    "action_id was reused with different arguments"
                )
            return {**copy.deepcopy(previous), "replayed": True}
        if self.faults.take("write_unknown_no_receipt"):
            raise ResultUnknownError("connection lost before a receipt existed")

        receipt = {
            "action_id": action_id,
            "ticket_id": str(payload["ticket_id"]),
            "recorded": True,
            "replayed": False,
            "request_fingerprint": request_fingerprint,
        }
        self.effects.append(
            {
                "action_id": action_id,
                "logical_operation": logical_operation,
                "payload": copy.deepcopy(payload),
                "fence": fence,
            }
        )
        self.receipts[action_id] = copy.deepcopy(receipt)
        if self.faults.take("write_committed_response_lost"):
            raise ResultUnknownError("write committed but response was lost")
        return receipt

    def lookup_receipt(self, action_id: str) -> dict[str, Any] | None:
        receipt = self.receipts.get(action_id)
        return copy.deepcopy(receipt) if receipt is not None else None

    @property
    def duplicate_side_effects(self) -> int:
        operations = [effect["logical_operation"] for effect in self.effects]
        return len(operations) - len(set(operations))


class DurableLoop:
    def __init__(
        self,
        *,
        run_store: JsonRunStore,
        model: ScriptedModelService,
        effects: TicketEffectStore,
        max_attempts: int = 3,
        base_backoff_ms: int = 100,
    ) -> None:
        self.run_store = run_store
        self.model = model
        self.effects = effects
        self.max_attempts = max_attempts
        self.base_backoff_ms = base_backoff_ms

    def start(self, case: DurableCase, *, worker_id: str) -> DurableRunState:
        state = DurableRunState(
            run_id=f"durable::{case.case_id}",
            case_id=case.case_id,
            strategy="durable-loop-v1",
            status="running",
        )
        self._emit(state, "run_started")
        self._acquire_lease(state, worker_id)
        self._checkpoint(state, "run_started")
        return self.drive(case, state)

    def resume(self, case: DurableCase, *, worker_id: str) -> DurableRunState:
        run_id = f"durable::{case.case_id}"
        state = self.run_store.load(run_id)
        if state.status in TERMINAL_STATES:
            raise ValueError(
                f"terminal run cannot resume without an explicit recovery action: "
                f"{state.status}"
            )
        self._emit(state, "run_rehydrated", previous_status=state.status)
        state.status = "running"
        self._acquire_lease(state, worker_id)
        self._checkpoint(state, "worker_resumed")
        return self.drive(case, state)

    def request_cancel(self, run_id: str, *, reason: str) -> DurableRunState:
        state = self.run_store.load(run_id)
        state.cancel_requested = True
        self._emit(state, "cancel_requested", reason=reason)
        self._checkpoint(state, "cancel_requested")
        return state

    def drive(
        self,
        case: DurableCase,
        state: DurableRunState,
        *,
        stop_after_model: bool = False,
    ) -> DurableRunState:
        if self._finish_cancel(state):
            return state

        if state.current_step == 0:
            if not self._run_model_step(case, state):
                return state
            if case.fault == "restart_after_model" and self.model.faults.take(
                "restart_after_model"
            ):
                self._emit(state, "process_crashed", after="model")
                self._checkpoint(state, "process_crashed")
                raise ProcessCrash("simulated crash after model checkpoint")
            if case.fault in {"cancel_at_human_wait", "stale_worker"}:
                state.status = "waiting_human"
                self._emit(state, "human_wait_started")
                self._checkpoint(state, "human_wait")
                return state
            if stop_after_model:
                return state

        if self._finish_cancel(state):
            return state
        if state.current_step == 1:
            if not self._run_write_step(case, state):
                return state

        state.status = "completed"
        state.failure_code = None
        state.failure_detail = None
        self._emit(state, "run_completed")
        self._checkpoint(state, "completed")
        return state

    def _run_model_step(
        self,
        case: DurableCase,
        state: DurableRunState,
    ) -> bool:
        while True:
            attempt = state.attempts.get("model", 0) + 1
            state.attempts["model"] = attempt
            self._emit(state, "model_started", attempt=attempt)
            try:
                output = self.model.decide(case)
            except TransientStepError as exc:
                if attempt >= self.max_attempts:
                    self._fail(state, "retry_exhausted", str(exc))
                    return False
                delay = self.base_backoff_ms * (2 ** (attempt - 1))
                state.next_retry_ms = state.logical_clock_ms + delay
                self._emit(
                    state,
                    "retry_scheduled",
                    step="model",
                    attempt=attempt,
                    delay_ms=delay,
                    error="transient",
                )
                self._checkpoint(state, "before_retry_wait")
                state.logical_clock_ms = state.next_retry_ms
                state.next_retry_ms = None
                self._emit(state, "retry_wait_elapsed", step="model")
                continue
            except PermanentStepError as exc:
                self._emit(
                    state,
                    "retry_suppressed",
                    step="model",
                    reason="permanent_error",
                )
                self._fail(state, "invalid_request", str(exc))
                return False

            state.step_outputs["model"] = output
            state.completed_steps.append("model")
            state.current_step = 1
            self._emit(state, "model_completed", attempt=attempt)
            self._checkpoint(state, "after_model")
            return True

    def _run_write_step(
        self,
        case: DurableCase,
        state: DurableRunState,
    ) -> bool:
        action_id = f"{state.run_id}::record-followup"
        logical_operation = f"followup::{case.case_id}"
        arguments = copy.deepcopy(state.step_outputs["model"]["arguments"])
        state.pending_action = {
            "action_id": action_id,
            "logical_operation": logical_operation,
            "arguments": arguments,
        }
        state.attempts["write"] = state.attempts.get("write", 0) + 1
        self._emit(
            state,
            "write_started",
            action_id=action_id,
            fence=state.lease_epoch,
        )
        self._checkpoint(state, "before_write")
        try:
            receipt = self.effects.record_followup(
                action_id=action_id,
                logical_operation=logical_operation,
                payload=arguments,
                fence=state.lease_epoch,
            )
        except ResultUnknownError as exc:
            self._emit(state, "write_result_unknown", action_id=action_id)
            receipt = self.effects.lookup_receipt(action_id)
            if receipt is None:
                state.status = "waiting_reconciliation"
                state.failure_code = "result_unknown"
                state.failure_detail = str(exc)
                self._emit(
                    state,
                    "reconciliation_required",
                    action_id=action_id,
                )
                self._checkpoint(state, "waiting_reconciliation")
                return False
            self._emit(state, "receipt_recovered", action_id=action_id)
        except StaleWorkerError as exc:
            self._emit(state, "stale_worker_rejected", error=str(exc))
            self._fail(state, "stale_worker", str(exc))
            return False
        except IdempotencyConflictError as exc:
            self._emit(state, "idempotency_conflict", error=str(exc))
            self._fail(state, "idempotency_conflict", str(exc))
            return False

        state.step_outputs["write"] = receipt
        state.completed_steps.append("write")
        state.pending_action = None
        state.current_step = 2
        self._emit(
            state,
            "write_completed",
            action_id=action_id,
            replayed=bool(receipt.get("replayed", False)),
        )
        self._checkpoint(state, "after_write")
        return True

    def _finish_cancel(self, state: DurableRunState) -> bool:
        if not state.cancel_requested:
            return False
        state.status = "cancelled"
        state.failure_code = "cancelled_by_user"
        state.failure_detail = "cancel persisted before the next side effect"
        self._emit(state, "run_cancelled")
        self._checkpoint(state, "cancelled")
        return True

    def _acquire_lease(self, state: DurableRunState, worker_id: str) -> None:
        state.lease_epoch += 1
        state.lease_owner = worker_id
        self.effects.activate_fence(state.lease_epoch)
        self._emit(
            state,
            "lease_acquired",
            worker=worker_id,
            epoch=state.lease_epoch,
        )

    def _fail(self, state: DurableRunState, code: str, detail: str) -> None:
        state.status = "failed"
        state.failure_code = code
        state.failure_detail = detail
        self._emit(state, "run_failed", code=code)
        self._checkpoint(state, "failed")

    def _checkpoint(self, state: DurableRunState, reason: str) -> None:
        state.checkpoint_count += 1
        self._emit(
            state,
            "checkpoint_saved",
            checkpoint=state.checkpoint_count,
            reason=reason,
        )
        self.run_store.save(state)

    @staticmethod
    def _emit(state: DurableRunState, event_type: str, **data: Any) -> None:
        state.events.append(
            {"seq": len(state.events) + 1, "type": event_type, **data}
        )


def load_durable_cases(path: Path) -> list[DurableCase]:
    cases: list[DurableCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        data = json.loads(line)
        case_id = str(data["id"])
        if case_id in seen:
            raise ValueError(
                f"duplicate durable case on line {line_number}: {case_id}"
            )
        seen.add(case_id)
        expected = dict(data["expected"])
        cases.append(
            DurableCase(
                case_id=case_id,
                fault=str(data["fault"]),
                expected_status=str(expected["status"]),
                expected_side_effects=int(expected["side_effects"]),
                expected_duplicate_side_effects=int(
                    expected["duplicate_side_effects"]
                ),
                expected_model_attempts=int(expected["model_attempts"]),
                expected_failure_code=(
                    str(expected["failure_code"])
                    if expected.get("failure_code") is not None
                    else None
                ),
                required_events=tuple(
                    str(item) for item in expected.get("required_events", [])
                ),
            )
        )
    if not cases:
        raise ValueError("durable case dataset must not be empty")
    return cases


def run_durable_eval(cases_path: Path, state_root: Path) -> DurableEvalResult:
    cases = load_durable_cases(cases_path)
    runs: list[DurableRun] = []
    for case in cases:
        runs.append(_run_baseline_case(case))
        runs.append(_run_durable_case(case, state_root / case.case_id))

    baseline_runs = [run for run in runs if run.strategy == "process-loop-v1"]
    candidate_runs = [run for run in runs if run.strategy == "durable-loop-v1"]
    baseline = _summarize("process-loop-v1", baseline_runs)
    candidate = _summarize("durable-loop-v1", candidate_runs)
    baseline_pass = {run.case.case_id: run.passed for run in baseline_runs}
    candidate_pass = {run.case.case_id: run.passed for run in candidate_runs}
    improvements = tuple(
        sorted(
            case_id
            for case_id, passed in baseline_pass.items()
            if not passed and candidate_pass[case_id]
        )
    )
    regressions = tuple(
        sorted(
            case_id
            for case_id, passed in baseline_pass.items()
            if passed and not candidate_pass[case_id]
        )
    )
    candidate_by_id = {run.case.case_id: run for run in candidate_runs}
    gate_checks = {
        "all_candidate_cases_pass": all(run.passed for run in candidate_runs),
        "at_least_one_measured_improvement": bool(improvements),
        "no_case_regressions": not regressions,
        "restart_rehydrates_checkpoint": _has_event(
            candidate_by_id["restart-after-model"], "run_rehydrated"
        ),
        "unknown_committed_write_recovers_receipt": _has_event(
            candidate_by_id["write-receipt-recovery"], "receipt_recovered"
        ),
        "unknown_unconfirmed_write_stops_for_reconciliation": (
            candidate_by_id["write-unknown"].state.status
            == "waiting_reconciliation"
            and candidate_by_id["write-unknown"].side_effect_count == 0
        ),
        "cancel_blocks_later_side_effect": (
            candidate_by_id["cancel-at-human-wait"].state.status == "cancelled"
            and candidate_by_id["cancel-at-human-wait"].side_effect_count == 0
        ),
        "stale_worker_is_fenced": _has_event(
            candidate_by_id["stale-worker"], "stale_worker_rejected"
        ),
        "candidate_has_no_duplicate_side_effect": (
            candidate.duplicate_side_effects == 0
        ),
        "candidate_has_no_blind_retry": candidate.blind_retries == 0,
    }
    return DurableEvalResult(
        version="0.6.0",
        baseline=baseline,
        candidate=candidate,
        improvements=improvements,
        regressions=regressions,
        gate_checks=gate_checks,
        gate_passed=all(gate_checks.values()),
        runs=tuple(runs),
    )


def _run_durable_case(case: DurableCase, state_root: Path) -> DurableRun:
    faults = FaultPlan(case.fault)
    model = ScriptedModelService(faults)
    effects = TicketEffectStore(faults)
    store = JsonRunStore(state_root)
    loop = DurableLoop(run_store=store, model=model, effects=effects)
    try:
        state = loop.start(case, worker_id="worker-a")
    except ProcessCrash:
        loop = DurableLoop(run_store=store, model=model, effects=effects)
        state = loop.resume(case, worker_id="worker-b")

    if case.fault == "cancel_at_human_wait":
        loop.request_cancel(state.run_id, reason="user cancelled during review")
        loop = DurableLoop(run_store=store, model=model, effects=effects)
        state = loop.resume(case, worker_id="worker-b")
    elif case.fault == "stale_worker":
        stale_epoch = state.lease_epoch
        loop = DurableLoop(run_store=store, model=model, effects=effects)
        resumed = store.load(state.run_id)
        loop._emit(resumed, "run_rehydrated", previous_status=resumed.status)
        loop._acquire_lease(resumed, "worker-b")
        loop._checkpoint(resumed, "lease_takeover")
        pending = resumed.step_outputs["model"]["arguments"]
        try:
            effects.record_followup(
                action_id=f"{resumed.run_id}::stale-attempt",
                logical_operation=f"followup::{case.case_id}",
                payload=pending,
                fence=stale_epoch,
            )
        except StaleWorkerError as exc:
            loop._emit(resumed, "stale_worker_rejected", error=str(exc))
            loop._checkpoint(resumed, "stale_worker_rejected")
        resumed.status = "running"
        state = loop.drive(case, resumed)

    return _make_run("durable-loop-v1", case, state, effects, blind_retries=0)


def _run_baseline_case(case: DurableCase) -> DurableRun:
    faults = FaultPlan(case.fault)
    model = ScriptedModelService(faults)
    effects = TicketEffectStore(faults)
    state = DurableRunState(
        run_id=f"process::{case.case_id}",
        case_id=case.case_id,
        strategy="process-loop-v1",
        status="running",
        lease_epoch=1,
    )
    effects.activate_fence(1)
    state.events.append({"seq": 1, "type": "run_started"})
    blind_retries = 0
    output: dict[str, Any] | None = None
    for attempt in range(1, 4):
        state.attempts["model"] = attempt
        state.events.append(
            {"seq": len(state.events) + 1, "type": "model_started", "attempt": attempt}
        )
        try:
            output = model.decide(case)
            break
        except (TransientStepError, PermanentStepError) as exc:
            if attempt < 3:
                blind_retries += 1
                state.events.append(
                    {
                        "seq": len(state.events) + 1,
                        "type": "blind_retry",
                        "step": "model",
                    }
                )
                continue
            state.status = "failed"
            state.failure_code = "retry_exhausted"
            state.failure_detail = str(exc)
            state.events.append(
                {"seq": len(state.events) + 1, "type": "run_failed"}
            )
            return _make_run(
                "process-loop-v1", case, state, effects, blind_retries
            )

    if output is None:
        raise AssertionError("baseline model loop ended without output")
    state.step_outputs["model"] = output
    state.current_step = 1
    if case.fault == "restart_after_model":
        state.events.append(
            {"seq": len(state.events) + 1, "type": "process_restarted_from_zero"}
        )
        state.attempts["model"] += 1
        output = model.decide(case)
        state.step_outputs["model"] = output
    if case.fault == "cancel_at_human_wait":
        state.events.append(
            {"seq": len(state.events) + 1, "type": "cancel_lost_with_process"}
        )
    arguments = copy.deepcopy(output["arguments"])
    logical_operation = f"followup::{case.case_id}"
    if case.fault == "stale_worker":
        for worker in ("worker-a", "worker-b"):
            effects.record_followup(
                action_id=f"{state.run_id}::{worker}",
                logical_operation=logical_operation,
                payload=arguments,
                fence=1,
            )
    else:
        for attempt in range(1, 4):
            state.attempts["write"] = attempt
            try:
                effects.record_followup(
                    action_id=f"{state.run_id}::attempt-{attempt}",
                    logical_operation=logical_operation,
                    payload=arguments,
                    fence=1,
                )
                break
            except ResultUnknownError:
                blind_retries += 1
                state.events.append(
                    {
                        "seq": len(state.events) + 1,
                        "type": "blind_retry",
                        "step": "write",
                    }
                )
    state.status = "completed"
    state.current_step = 2
    state.events.append(
        {"seq": len(state.events) + 1, "type": "run_completed"}
    )
    return _make_run("process-loop-v1", case, state, effects, blind_retries)


def _make_run(
    strategy: str,
    case: DurableCase,
    state: DurableRunState,
    effects: TicketEffectStore,
    blind_retries: int,
) -> DurableRun:
    event_types = [str(event["type"]) for event in state.events]
    missing = sorted(set(case.required_events) - set(event_types))
    grades = (
        DurableGrade(
            "expected_status",
            state.status == case.expected_status,
            f"actual={state.status}, expected={case.expected_status}",
        ),
        DurableGrade(
            "side_effect_count",
            len(effects.effects) == case.expected_side_effects,
            f"actual={len(effects.effects)}, expected={case.expected_side_effects}",
        ),
        DurableGrade(
            "duplicate_side_effects",
            effects.duplicate_side_effects
            == case.expected_duplicate_side_effects,
            (
                f"actual={effects.duplicate_side_effects}, "
                f"expected={case.expected_duplicate_side_effects}"
            ),
        ),
        DurableGrade(
            "model_attempts",
            state.attempts.get("model", 0) == case.expected_model_attempts,
            (
                f"actual={state.attempts.get('model', 0)}, "
                f"expected={case.expected_model_attempts}"
            ),
        ),
        DurableGrade(
            "failure_code",
            state.failure_code == case.expected_failure_code,
            (
                f"actual={state.failure_code}, "
                f"expected={case.expected_failure_code}"
            ),
        ),
        DurableGrade(
            "required_events",
            not missing,
            "all required events present" if not missing else f"missing={missing}",
        ),
    )
    return DurableRun(
        strategy=strategy,
        case=case,
        state=state,
        side_effect_count=len(effects.effects),
        duplicate_side_effects=effects.duplicate_side_effects,
        blind_retries=blind_retries,
        grades=grades,
        passed=all(grade.passed for grade in grades),
    )


def _summarize(strategy: str, runs: list[DurableRun]) -> DurableSummary:
    terminal = sum(run.state.status in TERMINAL_STATES for run in runs)
    return DurableSummary(
        strategy=strategy,
        cases=len(runs),
        passed_cases=sum(run.passed for run in runs),
        case_pass_rate=_rate(sum(run.passed for run in runs), len(runs)),
        total_model_attempts=sum(
            run.state.attempts.get("model", 0) for run in runs
        ),
        duplicate_side_effects=sum(run.duplicate_side_effects for run in runs),
        blind_retries=sum(run.blind_retries for run in runs),
        explicit_terminal_rate=_rate(terminal, len(runs)),
    )


def _has_event(run: DurableRun, event_type: str) -> bool:
    return any(event["type"] == event_type for event in run.state.events)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()
