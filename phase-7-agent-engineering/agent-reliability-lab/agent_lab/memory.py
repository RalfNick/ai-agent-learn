from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any


SENSITIVE_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
)

ALLOWED_EVIDENCE = {"explicit_user", "verified_reviewer"}
SOURCE_OF_TRUTH_EVIDENCE = {"business_record"}
AUTHORITY = {"verified_reviewer": 2, "explicit_user": 3}


@dataclass(frozen=True)
class MemoryCandidate:
    case_id: str
    namespace: tuple[str, ...]
    kind: str
    key: str
    statement: str
    evidence_type: str
    source_run_id: str
    source_ref: str
    reusable: bool
    sensitivity: str
    valid_until: str | None
    as_of: str


@dataclass(frozen=True)
class MemoryQuery:
    namespace: tuple[str, ...]
    kinds: tuple[str, ...]
    query_terms: tuple[str, ...]
    as_of: str
    case_id: str = "manual-recall"


@dataclass(frozen=True)
class MemorySelector:
    case_id: str
    namespace: tuple[str, ...]
    kind: str
    key: str
    as_of: str


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    namespace: tuple[str, ...]
    kind: str
    key: str
    statement: str | None
    provenance: dict[str, str]
    valid_until: str | None
    status: str
    version: int
    content_hash: str
    created_at: str
    created_by_case: str

    def to_dict(self, *, redact_statement: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["namespace"] = list(self.namespace)
        if redact_statement:
            data["statement"] = None
        return data


@dataclass(frozen=True)
class MemoryDecision:
    case_id: str
    action: str
    reason: str
    record: MemoryRecord | None = None
    affected_ids: tuple[str, ...] = ()
    recalled_ids: tuple[str, ...] = ()

    def to_dict(self, *, purge_ids: set[str] | None = None) -> dict[str, Any]:
        purge_ids = purge_ids or set()
        return {
            "case_id": self.case_id,
            "action": self.action,
            "reason": self.reason,
            "record": (
                self.record.to_dict(
                    redact_statement=self.record.memory_id in purge_ids
                )
                if self.record
                else None
            ),
            "affected_ids": list(self.affected_ids),
            "recalled_ids": list(self.recalled_ids),
        }


@dataclass(frozen=True)
class MemoryCase:
    case_id: str
    operation: str
    expected_action: str
    candidate: MemoryCandidate | None = None
    query: MemoryQuery | None = None
    selector: MemorySelector | None = None
    expected_recall_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryCaseResult:
    case_id: str
    operation: str
    expected_action: str
    decision: MemoryDecision
    matched: bool

    def to_dict(self, *, purge_ids: set[str]) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "operation": self.operation,
            "expected_action": self.expected_action,
            "decision": self.decision.to_dict(purge_ids=purge_ids),
            "matched": self.matched,
        }


@dataclass(frozen=True)
class MemoryReviewResult:
    version: str
    total_cases: int
    matched_cases: int
    decision_counts: dict[str, int]
    gate_checks: dict[str, bool]
    gate_passed: bool
    cases: tuple[MemoryCaseResult, ...]
    final_store: tuple[MemoryRecord, ...]

    @property
    def purge_ids(self) -> set[str]:
        return {
            record.memory_id
            for record in self.final_store
            if record.status == "deleted"
        }

    def summary_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "comparison_scope": (
                "deterministic memory-policy fixtures; not a model-quality "
                "or retrieval benchmark"
            ),
            "total_cases": self.total_cases,
            "matched_cases": self.matched_cases,
            "decision_counts": self.decision_counts,
            "gate_checks": self.gate_checks,
            "gate_passed": self.gate_passed,
        }

    def to_dict(self) -> dict[str, Any]:
        purge_ids = self.purge_ids
        return {
            **self.summary_dict(),
            "cases": [
                case.to_dict(purge_ids=purge_ids) for case in self.cases
            ],
            "final_store": [record.to_dict() for record in self.final_store],
        }


class MemoryStore:
    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records)

    def active_records(self) -> tuple[MemoryRecord, ...]:
        return tuple(record for record in self._records if record.status == "active")

    def records_for(
        self,
        namespace: tuple[str, ...],
        kind: str,
        key: str,
    ) -> tuple[MemoryRecord, ...]:
        return tuple(
            record
            for record in self._records
            if record.namespace == namespace
            and record.kind == kind
            and record.key == key
        )

    def apply(self, decision: MemoryDecision) -> None:
        if decision.action not in {"store", "supersede"} or decision.record is None:
            return
        if decision.action == "supersede":
            self._records = [
                replace(record, status="superseded")
                if record.namespace == decision.record.namespace
                and record.kind == decision.record.kind
                and record.key == decision.record.key
                and record.status == "active"
                else record
                for record in self._records
            ]
        self._records.append(decision.record)

    def recall(self, query: MemoryQuery) -> tuple[MemoryRecord, ...]:
        terms = {_normalize_token(term) for term in query.query_terms if term.strip()}
        candidates = (
            record
            for record in self._records
            if record.namespace == query.namespace
        )
        recalled: list[tuple[int, MemoryRecord]] = []
        for record in candidates:
            if record.kind not in query.kinds or record.status != "active":
                continue
            if record.valid_until and _parse_date(record.valid_until) < _parse_date(
                query.as_of
            ):
                continue
            searchable = _tokens(f"{record.key} {record.statement or ''}")
            score = len(terms & searchable)
            if terms and score == 0:
                continue
            recalled.append((score, record))
        recalled.sort(key=lambda item: (-item[0], item[1].memory_id))
        return tuple(record for _, record in recalled)

    def delete(self, selector: MemorySelector) -> MemoryDecision:
        affected: list[str] = []
        updated: list[MemoryRecord] = []
        for record in self._records:
            if (
                record.namespace == selector.namespace
                and record.kind == selector.kind
                and record.key == selector.key
            ):
                affected.append(record.memory_id)
                updated.append(
                    replace(
                        record,
                        statement=None,
                        status="deleted",
                        content_hash="purged",
                        provenance={
                            "source": "deletion_request",
                            "source_run_id": selector.case_id,
                            "source_ref": "content_purged",
                        },
                    )
                )
            else:
                updated.append(record)
        self._records = updated
        return MemoryDecision(
            case_id=selector.case_id,
            action="delete" if affected else "reject",
            reason="content_purged_tombstone_kept" if affected else "memory_not_found",
            affected_ids=tuple(affected),
        )


def load_memory_cases(path: Path) -> tuple[MemoryCase, ...]:
    cases: list[MemoryCase] = []
    seen_case_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        data = json.loads(raw_line)
        try:
            case_id = str(data["id"])
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate memory case id {case_id!r}")
            seen_case_ids.add(case_id)
            operation = str(data["operation"])
            expected_action = str(data["expected_action"])
            if operation == "candidate":
                candidate = MemoryCandidate(
                    case_id=case_id,
                    namespace=tuple(str(item) for item in data["namespace"]),
                    kind=str(data["kind"]),
                    key=str(data["key"]),
                    statement=str(data["statement"]),
                    evidence_type=str(data["evidence_type"]),
                    source_run_id=str(data["source_run_id"]),
                    source_ref=str(data["source_ref"]),
                    reusable=bool(data["reusable"]),
                    sensitivity=str(data["sensitivity"]),
                    valid_until=(
                        str(data["valid_until"])
                        if data.get("valid_until") is not None
                        else None
                    ),
                    as_of=str(data["as_of"]),
                )
                cases.append(
                    MemoryCase(case_id, operation, expected_action, candidate=candidate)
                )
            elif operation == "recall":
                query = MemoryQuery(
                    case_id=case_id,
                    namespace=tuple(str(item) for item in data["namespace"]),
                    kinds=tuple(str(item) for item in data["kinds"]),
                    query_terms=tuple(str(item) for item in data["query_terms"]),
                    as_of=str(data["as_of"]),
                )
                cases.append(
                    MemoryCase(
                        case_id,
                        operation,
                        expected_action,
                        query=query,
                        expected_recall_ids=tuple(
                            str(item) for item in data.get("expected_recall_ids", [])
                        ),
                    )
                )
            elif operation == "delete":
                selector = MemorySelector(
                    case_id=case_id,
                    namespace=tuple(str(item) for item in data["namespace"]),
                    kind=str(data["kind"]),
                    key=str(data["key"]),
                    as_of=str(data["as_of"]),
                )
                cases.append(
                    MemoryCase(case_id, operation, expected_action, selector=selector)
                )
            else:
                raise ValueError(f"unsupported memory operation {operation!r}")
        except KeyError as exc:
            raise ValueError(
                f"memory case line {line_number} is missing {exc.args[0]}"
            ) from exc
    return tuple(cases)


def evaluate_candidate(
    candidate: MemoryCandidate | None,
    store: MemoryStore,
) -> MemoryDecision:
    if candidate is None:
        raise ValueError("candidate is required")
    if (
        candidate.sensitivity == "secret"
        or _contains_sensitive_value(candidate.statement)
    ):
        return _decision(candidate, "reject", "sensitive_or_prohibited")
    if (
        candidate.evidence_type in SOURCE_OF_TRUTH_EVIDENCE
        or candidate.kind == "case_fact"
    ):
        return _decision(candidate, "route_to_source", "business_source_of_truth")
    if not candidate.reusable:
        return _decision(candidate, "reject", "not_reusable_across_tasks")
    if candidate.evidence_type == "model_inference":
        return _decision(candidate, "reject", "inference_requires_confirmation")
    if len(candidate.namespace) < 2 or any(not part for part in candidate.namespace):
        return _decision(candidate, "reject", "invalid_namespace")
    if candidate.evidence_type not in ALLOWED_EVIDENCE:
        return _decision(candidate, "reject", "unsupported_evidence")
    if candidate.valid_until and _parse_date(candidate.valid_until) < _parse_date(
        candidate.as_of
    ):
        return _decision(candidate, "reject", "already_expired")

    existing = [
        record
        for record in store.records_for(
            candidate.namespace, candidate.kind, candidate.key
        )
        if record.status == "active"
    ]
    version = max((record.version for record in existing), default=0) + 1
    action = "store"
    reason = "policy_accepted"
    if existing:
        current_authority = max(
            AUTHORITY.get(record.provenance.get("source", ""), 0)
            for record in existing
        )
        if AUTHORITY[candidate.evidence_type] < current_authority:
            return _decision(candidate, "reject", "lower_authority_conflict")
        action = "supersede"
        reason = "newer_equal_or_higher_authority"
    record = _record(candidate, version)
    return MemoryDecision(
        case_id=candidate.case_id,
        action=action,
        reason=reason,
        record=record,
        affected_ids=tuple(record.memory_id for record in existing),
    )


def run_memory_review(path: Path) -> MemoryReviewResult:
    store = MemoryStore()
    results: list[MemoryCaseResult] = []
    for case in load_memory_cases(path):
        if case.operation == "candidate":
            decision = evaluate_candidate(case.candidate, store)
            store.apply(decision)
        elif case.operation == "recall":
            if case.query is None:
                raise ValueError(f"recall case {case.case_id} lacks a query")
            recalled = store.recall(case.query)
            decision = MemoryDecision(
                case_id=case.case_id,
                action="store" if recalled else "reject",
                reason="recalled_active_memory" if recalled else "namespace_first_no_match",
                recalled_ids=tuple(record.memory_id for record in recalled),
            )
        elif case.operation == "delete":
            if case.selector is None:
                raise ValueError(f"delete case {case.case_id} lacks a selector")
            decision = store.delete(case.selector)
        else:
            raise ValueError(f"unsupported memory operation {case.operation!r}")
        matched = decision.action == case.expected_action
        if case.operation == "recall":
            matched = matched and decision.recalled_ids == case.expected_recall_ids
        results.append(
            MemoryCaseResult(
                case_id=case.case_id,
                operation=case.operation,
                expected_action=case.expected_action,
                decision=decision,
                matched=matched,
            )
        )

    counts = {action: 0 for action in (
        "store", "route_to_source", "reject", "supersede", "delete"
    )}
    for result in results:
        counts[result.decision.action] = counts.get(result.decision.action, 0) + 1

    final_records = store.records
    gate_checks = {
        "expected_decisions_matched": all(result.matched for result in results),
        "required_fields_present": all(
            record.namespace
            and record.kind
            and record.provenance
            and record.status in {"active", "superseded", "deleted"}
            for record in final_records
        ),
        "sensitive_values_absent": all(
            not _contains_sensitive_value(record.statement or "")
            for record in final_records
        ),
        "namespace_isolation_enforced": not next(
            result
            for result in results
            if result.case_id == "cross-tenant-recall"
        ).decision.recalled_ids,
        "one_active_version_per_key": _one_active_version_per_key(final_records),
        "deleted_content_purged": all(
            record.statement is None
            for record in final_records
            if record.status == "deleted"
        ),
    }
    return MemoryReviewResult(
        version="0.8.0",
        total_cases=len(results),
        matched_cases=sum(result.matched for result in results),
        decision_counts=counts,
        gate_checks=gate_checks,
        gate_passed=all(gate_checks.values()),
        cases=tuple(results),
        final_store=final_records,
    )


def _decision(
    candidate: MemoryCandidate,
    action: str,
    reason: str,
) -> MemoryDecision:
    return MemoryDecision(candidate.case_id, action, reason)


def _record(candidate: MemoryCandidate, version: int) -> MemoryRecord:
    content_hash = "sha256:" + hashlib.sha256(
        candidate.statement.encode("utf-8")
    ).hexdigest()
    identity = "/".join(
        (*candidate.namespace, candidate.kind, candidate.key, str(version))
    )
    memory_id = "mem_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return MemoryRecord(
        memory_id=memory_id,
        namespace=candidate.namespace,
        kind=candidate.kind,
        key=candidate.key,
        statement=candidate.statement,
        provenance={
            "source": candidate.evidence_type,
            "source_run_id": candidate.source_run_id,
            "source_ref": candidate.source_ref,
        },
        valid_until=candidate.valid_until,
        status="active",
        version=version,
        content_hash=content_hash,
        created_at=candidate.as_of,
        created_by_case=candidate.case_id,
    )


def _one_active_version_per_key(records: tuple[MemoryRecord, ...]) -> bool:
    active: set[tuple[tuple[str, ...], str, str]] = set()
    for record in records:
        if record.status != "active":
            continue
        identity = (record.namespace, record.kind, record.key)
        if identity in active:
            return False
        active.add(identity)
    return True


def _tokens(value: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in re.findall(r"[A-Za-z0-9_]+", value)
        if token
    }


def _normalize_token(value: str) -> str:
    return value.strip().lower()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _contains_sensitive_value(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return any(pattern.search(serialized) for pattern in SENSITIVE_PATTERNS)
