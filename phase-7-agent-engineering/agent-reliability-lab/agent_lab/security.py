from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolPolicy:
    risk: str
    mutating: bool
    approval_required: bool
    required_role: str | None = None
    max_auth_age_seconds: int | None = None


TOOL_POLICIES: dict[str, ToolPolicy] = {
    "lookup_ticket": ToolPolicy("read", False, False),
    "update_ticket_label": ToolPolicy("reversible", True, False),
    "send_customer_message": ToolPolicy(
        "external", True, True, required_role="reviewer"
    ),
    "deploy_production": ToolPolicy(
        "critical",
        True,
        True,
        required_role="release_manager",
        max_auth_age_seconds=300,
    ),
}


@dataclass(frozen=True)
class ActionProposal:
    action_id: str
    tool: str
    resource: str
    arguments: dict[str, Any]
    environment: str
    requester: str
    rollback: dict[str, Any] | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    decision: str
    reviewer_id: str
    reviewer_role: str
    decided_at: int
    auth_age_seconds: int


@dataclass(frozen=True)
class ApprovalEnvelope:
    approval_id: str
    action_id: str
    tool: str
    resource: str
    environment: str
    risk: str
    requester: str
    arguments_hash: str
    action_fingerprint: str
    requested_at: int
    expires_at: int
    required_role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityRunState:
    action: ActionProposal
    risk: str
    status: str
    reason: str
    approval: ApprovalEnvelope | None = None
    receipt: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        rollback = None
        if self.rollback is not None:
            rollback = {
                "tool": str(self.rollback.get("tool", "")),
                "arguments_hash": _hash_value(self.rollback.get("arguments", {})),
            }
        return {
            "action_id": self.action.action_id,
            "tool": self.action.tool,
            "resource": self.action.resource,
            "environment": self.action.environment,
            "risk": self.risk,
            "arguments_hash": _hash_value(self.action.arguments),
            "status": self.status,
            "reason": self.reason,
            "approval": self.approval.to_dict() if self.approval else None,
            "receipt": dict(self.receipt) if self.receipt else None,
            "rollback": rollback,
            "events": [dict(event) for event in self.events],
        }


@dataclass(frozen=True)
class SecurityCase:
    case_id: str
    now: int
    approval_ttl: int
    action: ActionProposal
    decision: ApprovalDecision | None
    resume_action: ActionProposal | None
    resume_count: int
    expected_status: str
    expected_reason: str
    expected_effects: int


@dataclass(frozen=True)
class SecurityCaseResult:
    case_id: str
    expected_status: str
    expected_reason: str
    expected_effects: int
    status: str
    reason: str
    risk: str
    mutation_count: int
    state: SecurityRunState
    matched: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "expected_status": self.expected_status,
            "expected_reason": self.expected_reason,
            "expected_effects": self.expected_effects,
            "status": self.status,
            "reason": self.reason,
            "risk": self.risk,
            "mutation_count": self.mutation_count,
            "state": self.state.to_dict(),
            "matched": self.matched,
        }


@dataclass(frozen=True)
class SecurityEvalResult:
    version: str
    total_cases: int
    matched_cases: int
    status_counts: dict[str, int]
    gate_checks: dict[str, bool]
    gate_passed: bool
    cases: tuple[SecurityCaseResult, ...]

    def case_by_id(self, case_id: str) -> SecurityCaseResult:
        return next(case for case in self.cases if case.case_id == case_id)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "comparison_scope": (
                "deterministic human-control fixtures; not a penetration test, "
                "IAM audit, or model-safety benchmark"
            ),
            "total_cases": self.total_cases,
            "matched_cases": self.matched_cases,
            "status_counts": self.status_counts,
            "gate_checks": self.gate_checks,
            "gate_passed": self.gate_passed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }


class EffectStore:
    def __init__(self) -> None:
        self.mutation_count = 0
        self.receipts: dict[str, dict[str, Any]] = {}

    def execute(
        self,
        action: ActionProposal,
        policy: ToolPolicy,
        *,
        credential: str | None,
    ) -> dict[str, Any]:
        fingerprint = action_fingerprint(action)
        existing = self.receipts.get(action.action_id)
        if existing is not None:
            if existing["fingerprint"] != fingerprint:
                raise ValueError("action_changed")
            return {
                "receipt_id": existing["receipt_id"],
                "replayed": True,
                "mutating": existing["mutating"],
            }

        if policy.mutating and not credential:
            raise ValueError("credential_unavailable")
        if policy.mutating:
            self.mutation_count += 1
        receipt = {
            "receipt_id": f"rcpt-{fingerprint[:12]}",
            "fingerprint": fingerprint,
            "mutating": policy.mutating,
        }
        self.receipts[action.action_id] = receipt
        return {
            "receipt_id": receipt["receipt_id"],
            "replayed": False,
            "mutating": policy.mutating,
        }


class JsonApprovalStore:
    """Trusted checkpoint storage; public reports use SecurityRunState.to_dict."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state: SecurityRunState) -> None:
        target = self._path(state.action.action_id)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(_checkpoint_dict(state), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def load(self, action_id: str) -> SecurityRunState:
        state = _state_from_checkpoint(
            json.loads(self._path(action_id).read_text(encoding="utf-8"))
        )
        if state.action.action_id != action_id:
            raise ValueError("checkpoint action id mismatch")
        if (
            state.approval is not None
            and action_fingerprint(state.action)
            != state.approval.action_fingerprint
        ):
            raise ValueError("checkpoint action fingerprint mismatch")
        return state

    def _path(self, action_id: str) -> Path:
        safe_id = hashlib.sha256(action_id.encode("utf-8")).hexdigest()
        return self.root / f"{safe_id}.json"


class SecurityEngine:
    def __init__(
        self,
        *,
        credentials: dict[str, str] | None = None,
        effects: EffectStore | None = None,
    ) -> None:
        self.credentials = credentials or {
            "update_ticket_label": "fixture-ticket-write-token",
            "send_customer_message": "fixture-message-send-token",
            "deploy_production": "fixture-deploy-token",
        }
        self.effects = effects or EffectStore()

    def propose(
        self,
        action: ActionProposal,
        *,
        now: int,
        approval_ttl: int = 300,
    ) -> SecurityRunState:
        policy = TOOL_POLICIES.get(action.tool)
        if policy is None:
            return self._initial_state(action, "unknown", "denied", "unknown_tool")

        state = self._initial_state(action, policy.risk, "proposed", "classified")
        if policy.risk == "reversible":
            if action.rollback is None:
                return self._stop(state, "denied", "rollback_required")
            if not _valid_rollback(action):
                return self._stop(state, "denied", "invalid_rollback")

        if policy.approval_required:
            fingerprint = action_fingerprint(action)
            state.approval = ApprovalEnvelope(
                approval_id=f"approval-{fingerprint[:12]}",
                action_id=action.action_id,
                tool=action.tool,
                resource=action.resource,
                environment=action.environment,
                risk=policy.risk,
                requester=action.requester,
                arguments_hash=_hash_value(action.arguments),
                action_fingerprint=fingerprint,
                requested_at=now,
                expires_at=now + approval_ttl,
                required_role=str(policy.required_role),
            )
            state.status = "waiting_approval"
            state.reason = "approval_required"
            state.events.append(
                {
                    "sequence": len(state.events) + 1,
                    "type": "approval_requested",
                    "approval_id": state.approval.approval_id,
                    "expires_at": state.approval.expires_at,
                    "required_role": state.approval.required_role,
                }
            )
            return state

        reason = "auto_allowed" if policy.risk == "read" else "policy_allowed"
        return self._execute(state, policy, reason=reason)

    def resume(
        self,
        state: SecurityRunState,
        decision: ApprovalDecision,
        *,
        current_action: ActionProposal | None = None,
    ) -> SecurityRunState:
        action = current_action or state.action
        if state.status == "completed" and state.receipt is not None:
            policy = TOOL_POLICIES[state.action.tool]
            receipt = self.effects.execute(
                state.action,
                policy,
                credential=self.credentials.get(state.action.tool),
            )
            state.receipt = receipt
            state.events.append(
                {
                    "sequence": len(state.events) + 1,
                    "type": "receipt_replayed",
                    "receipt_id": receipt["receipt_id"],
                }
            )
            return state

        if state.status != "waiting_approval" or state.approval is None:
            raise ValueError(f"cannot resume state: {state.status}")

        state.events.append(
            {
                "sequence": len(state.events) + 1,
                "type": "approval_decided",
                "decision": decision.decision,
                "reviewer_id": decision.reviewer_id,
                "reviewer_role": decision.reviewer_role,
                "decided_at": decision.decided_at,
            }
        )
        if decision.decision == "reject":
            return self._stop(state, "rejected", "rejected_by_reviewer")
        if decision.decision == "cancel":
            return self._stop(state, "cancelled", "cancelled_by_user")
        if decision.decision != "approve":
            return self._stop(state, "denied", "invalid_decision")

        if action_fingerprint(action) != state.approval.action_fingerprint:
            return self._stop(state, "blocked", "action_changed")
        if decision.decided_at < state.approval.requested_at:
            return self._stop(state, "denied", "invalid_approval_time")
        if decision.auth_age_seconds < 0:
            return self._stop(state, "denied", "invalid_auth_age")
        if decision.reviewer_id == state.approval.requester:
            return self._stop(state, "denied", "self_approval_not_allowed")
        if decision.decided_at > state.approval.expires_at:
            return self._stop(state, "expired", "approval_expired")
        if decision.reviewer_role != state.approval.required_role:
            return self._stop(state, "denied", "reviewer_not_authorized")

        policy = TOOL_POLICIES.get(action.tool)
        if policy is None or policy.risk != state.approval.risk:
            return self._stop(state, "blocked", "policy_changed")
        if (
            policy.max_auth_age_seconds is not None
            and decision.auth_age_seconds > policy.max_auth_age_seconds
        ):
            return self._stop(state, "denied", "authentication_stale")

        state.action = action
        return self._execute(state, policy, reason="approved")

    def _initial_state(
        self,
        action: ActionProposal,
        risk: str,
        status: str,
        reason: str,
    ) -> SecurityRunState:
        state = SecurityRunState(
            action=action,
            risk=risk,
            status=status,
            reason=reason,
            rollback=action.rollback,
        )
        state.events.append(
            {
                "sequence": 1,
                "type": "action_classified",
                "action_id": action.action_id,
                "tool": action.tool,
                "resource": action.resource,
                "environment": action.environment,
                "risk": risk,
                "arguments_hash": _hash_value(action.arguments),
                "argument_keys": sorted(action.arguments),
            }
        )
        return state

    def _execute(
        self,
        state: SecurityRunState,
        policy: ToolPolicy,
        *,
        reason: str,
    ) -> SecurityRunState:
        receipt = self.effects.execute(
            state.action,
            policy,
            credential=self.credentials.get(state.action.tool),
        )
        state.receipt = receipt
        state.status = "completed"
        state.reason = reason
        state.events.append(
            {
                "sequence": len(state.events) + 1,
                "type": "effect_executed" if policy.mutating else "read_executed",
                "receipt_id": receipt["receipt_id"],
                "mutating": policy.mutating,
            }
        )
        return state

    @staticmethod
    def _stop(
        state: SecurityRunState,
        status: str,
        reason: str,
    ) -> SecurityRunState:
        state.status = status
        state.reason = reason
        state.events.append(
            {
                "sequence": len(state.events) + 1,
                "type": "execution_stopped",
                "status": status,
                "reason": reason,
            }
        )
        return state


def action_fingerprint(action: ActionProposal) -> str:
    return _hash_value(
        {
            "action_id": action.action_id,
            "tool": action.tool,
            "resource": action.resource,
            "arguments": action.arguments,
            "environment": action.environment,
            "requester": action.requester,
        }
    )


def load_security_cases(path: Path) -> tuple[SecurityCase, ...]:
    cases: list[SecurityCase] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        data = json.loads(raw_line)
        case_id = str(data["id"])
        if case_id in seen:
            raise ValueError(f"duplicate security case id: {case_id}")
        seen.add(case_id)
        cases.append(
            SecurityCase(
                case_id=case_id,
                now=int(data["now"]),
                approval_ttl=int(data.get("approval_ttl", 300)),
                action=_load_action(data["action"]),
                decision=(
                    _load_decision(data["decision"])
                    if data.get("decision") is not None
                    else None
                ),
                resume_action=(
                    _load_action(data["resume_action"])
                    if data.get("resume_action") is not None
                    else None
                ),
                resume_count=int(data.get("resume_count", 1)),
                expected_status=str(data["expected_status"]),
                expected_reason=str(data["expected_reason"]),
                expected_effects=int(data["expected_effects"]),
            )
        )
    return tuple(cases)


def run_security_eval(path: Path) -> SecurityEvalResult:
    results = tuple(_run_case(case) for case in load_security_cases(path))
    by_id = {case.case_id: case for case in results}
    status_counts: dict[str, int] = {}
    for case in results:
        status_counts[case.status] = status_counts.get(case.status, 0) + 1

    no_effect_ids = (
        "external-rejected",
        "approval-expired",
        "arguments-changed-after-approval",
        "critical-wrong-reviewer",
    )
    gate_checks = {
        "expected_results": all(case.matched for case in results),
        "read_only_auto": by_id["read-only-auto"].reason == "auto_allowed",
        "reversible_has_rollback": (
            by_id["reversible-policy-allowed"].state.rollback is not None
        ),
        "approval_precedes_effect": _approval_precedes_effect(
            by_id["external-approved-once"]
        ),
        "rejection_expiry_and_drift_have_no_effect": all(
            by_id[case_id].mutation_count == 0 for case_id in no_effect_ids
        ),
        "critical_role_enforced": (
            by_id["critical-wrong-reviewer"].reason
            == "reviewer_not_authorized"
        ),
        "duplicate_resume_is_idempotent": (
            by_id["duplicate-approved-resume"].mutation_count == 1
            and any(
                event["type"] == "receipt_replayed"
                for event in by_id["duplicate-approved-resume"].state.events
            )
        ),
        "audit_is_redacted": all(_audit_is_redacted(case) for case in results),
    }
    return SecurityEvalResult(
        version="human-control-v1",
        total_cases=len(results),
        matched_cases=sum(case.matched for case in results),
        status_counts=status_counts,
        gate_checks=gate_checks,
        gate_passed=all(gate_checks.values()),
        cases=results,
    )


def _run_case(case: SecurityCase) -> SecurityCaseResult:
    engine = SecurityEngine()
    state = engine.propose(
        case.action,
        now=case.now,
        approval_ttl=case.approval_ttl,
    )
    if case.decision is not None:
        for _ in range(case.resume_count):
            state = engine.resume(
                state,
                case.decision,
                current_action=case.resume_action,
            )
    matched = (
        state.status == case.expected_status
        and state.reason == case.expected_reason
        and engine.effects.mutation_count == case.expected_effects
    )
    return SecurityCaseResult(
        case_id=case.case_id,
        expected_status=case.expected_status,
        expected_reason=case.expected_reason,
        expected_effects=case.expected_effects,
        status=state.status,
        reason=state.reason,
        risk=state.risk,
        mutation_count=engine.effects.mutation_count,
        state=state,
        matched=matched,
    )


def _load_action(data: dict[str, Any]) -> ActionProposal:
    arguments = data.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("action arguments must be an object")
    rollback = data.get("rollback")
    if rollback is not None and not isinstance(rollback, dict):
        raise ValueError("rollback must be an object")
    return ActionProposal(
        action_id=str(data["action_id"]),
        tool=str(data["tool"]),
        resource=str(data["resource"]),
        arguments=dict(arguments),
        environment=str(data["environment"]),
        requester=str(data["requester"]),
        rollback=dict(rollback) if rollback is not None else None,
    )


def _load_decision(data: dict[str, Any]) -> ApprovalDecision:
    return ApprovalDecision(
        decision=str(data["type"]),
        reviewer_id=str(data["reviewer_id"]),
        reviewer_role=str(data["reviewer_role"]),
        decided_at=int(data["at"]),
        auth_age_seconds=int(data["auth_age_seconds"]),
    )


def _checkpoint_dict(state: SecurityRunState) -> dict[str, Any]:
    return {
        "action": asdict(state.action),
        "risk": state.risk,
        "status": state.status,
        "reason": state.reason,
        "approval": state.approval.to_dict() if state.approval else None,
        "receipt": state.receipt,
        "rollback": state.rollback,
        "events": state.events,
    }


def _state_from_checkpoint(data: dict[str, Any]) -> SecurityRunState:
    approval_data = data.get("approval")
    return SecurityRunState(
        action=_load_action(data["action"]),
        risk=str(data["risk"]),
        status=str(data["status"]),
        reason=str(data["reason"]),
        approval=(
            ApprovalEnvelope(
                approval_id=str(approval_data["approval_id"]),
                action_id=str(approval_data["action_id"]),
                tool=str(approval_data["tool"]),
                resource=str(approval_data["resource"]),
                environment=str(approval_data["environment"]),
                risk=str(approval_data["risk"]),
                requester=str(approval_data["requester"]),
                arguments_hash=str(approval_data["arguments_hash"]),
                action_fingerprint=str(approval_data["action_fingerprint"]),
                requested_at=int(approval_data["requested_at"]),
                expires_at=int(approval_data["expires_at"]),
                required_role=str(approval_data["required_role"]),
            )
            if approval_data is not None
            else None
        ),
        receipt=(dict(data["receipt"]) if data.get("receipt") else None),
        rollback=(dict(data["rollback"]) if data.get("rollback") else None),
        events=[dict(event) for event in data.get("events", [])],
    )


def _hash_value(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _valid_rollback(action: ActionProposal) -> bool:
    if action.rollback is None:
        return False
    if action.tool != "update_ticket_label":
        return False
    if action.rollback.get("tool") != "restore_ticket_labels":
        return False
    rollback_arguments = action.rollback.get("arguments")
    if not isinstance(rollback_arguments, dict):
        return False
    return (
        rollback_arguments.get("ticket_id") == action.arguments.get("ticket_id")
        and action.resource == f"ticket:{action.arguments.get('ticket_id')}"
    )


def _approval_precedes_effect(case: SecurityCaseResult) -> bool:
    types = [event["type"] for event in case.state.events]
    return (
        "approval_decided" in types
        and "effect_executed" in types
        and types.index("approval_decided") < types.index("effect_executed")
    )


def _audit_is_redacted(case: SecurityCaseResult) -> bool:
    exported = json.dumps(case.to_dict(), ensure_ascii=False).lower()
    forbidden_values = (
        "original reviewed message",
        "changed after approval",
        "fixture-message-send-token",
        "fixture-deploy-token",
        "fixture-ticket-write-token",
    )
    return not any(value in exported for value in forbidden_values)
