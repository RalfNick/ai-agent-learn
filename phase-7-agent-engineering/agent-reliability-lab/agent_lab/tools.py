from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ToolError:
    code: str
    category: str
    message: str
    retryable: bool
    details: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return {
            "code": self.code,
            "category": self.category,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: JsonObject | None = None
    error: ToolError | None = None
    side_effects: tuple[str, ...] = ()
    replayed: bool = False

    def to_dict(self) -> JsonObject:
        return {
            "ok": self.ok,
            "output": self.output,
            "error": self.error.to_dict() if self.error else None,
            "side_effects": list(self.side_effects),
            "replayed": self.replayed,
        }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JsonObject
    output_schema: JsonObject
    required_permission: str
    effect: str = "read"
    approval: str = "never"
    idempotency_key: str | None = None
    retry_policy: str = "never"
    timeout_ms: int = 500

    def model_schema(self) -> JsonObject:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "strict": True,
        }

    def runtime_contract(self) -> JsonObject:
        return {
            "required_permission": self.required_permission,
            "effect": self.effect,
            "approval": self.approval,
            "idempotency_key": self.idempotency_key,
            "retry_policy": self.retry_policy,
            "timeout_ms": self.timeout_ms,
            "output_schema": self.output_schema,
        }


@dataclass(frozen=True)
class ProposedCall:
    name: str
    arguments: JsonObject

    @classmethod
    def from_dict(cls, payload: JsonObject) -> "ProposedCall":
        return cls(name=payload["name"], arguments=payload.get("arguments", {}))


@dataclass(frozen=True)
class ToolCase:
    case_id: str
    intent: str
    actor_permissions: frozenset[str]
    approved: bool
    baseline_calls: tuple[ProposedCall, ...]
    candidate_calls: tuple[ProposedCall, ...]
    expected: JsonObject

    @classmethod
    def from_dict(cls, payload: JsonObject) -> "ToolCase":
        return cls(
            case_id=payload["case_id"],
            intent=payload["intent"],
            actor_permissions=frozenset(payload["actor_permissions"]),
            approved=payload.get("approved", False),
            baseline_calls=tuple(
                ProposedCall.from_dict(item) for item in payload["baseline_calls"]
            ),
            candidate_calls=tuple(
                ProposedCall.from_dict(item) for item in payload["candidate_calls"]
            ),
            expected=payload["expected"],
        )


@dataclass(frozen=True)
class ToolGrade:
    name: str
    passed: bool
    expected: Any
    actual: Any

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class ToolRun:
    strategy: str
    case: ToolCase
    results: tuple[ToolResult, ...]
    grades: tuple[ToolGrade, ...]
    side_effect_count: int

    @property
    def passed(self) -> bool:
        return all(grade.passed for grade in self.grades)

    def to_dict(self) -> JsonObject:
        return {
            "strategy": self.strategy,
            "case_id": self.case.case_id,
            "intent": self.case.intent,
            "passed": self.passed,
            "side_effect_count": self.side_effect_count,
            "results": [result.to_dict() for result in self.results],
            "grades": [grade.to_dict() for grade in self.grades],
        }


@dataclass(frozen=True)
class ToolSummary:
    strategy: str
    total_cases: int
    passed_cases: int
    unsafe_side_effects: int
    duplicate_side_effects: int
    structured_error_rate: float
    model_schema_bytes: int

    @property
    def case_pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases else 0.0

    def to_dict(self) -> JsonObject:
        return {
            "strategy": self.strategy,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "case_pass_rate": self.case_pass_rate,
            "unsafe_side_effects": self.unsafe_side_effects,
            "duplicate_side_effects": self.duplicate_side_effects,
            "structured_error_rate": self.structured_error_rate,
            "model_schema_bytes": self.model_schema_bytes,
        }


@dataclass(frozen=True)
class ToolEvalResult:
    version: str
    baseline: ToolSummary
    candidate: ToolSummary
    runs: tuple[ToolRun, ...]
    improvements: tuple[str, ...]
    regressions: tuple[str, ...]
    gate_checks: JsonObject

    @property
    def gate_passed(self) -> bool:
        return all(self.gate_checks.values())

    def summary_dict(self) -> JsonObject:
        return {
            "version": self.version,
            "scope": (
                "Deterministic tool-contract conformance over fixed proposed "
                "calls; not a model tool-selection benchmark."
            ),
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "improvements": list(self.improvements),
            "regressions": list(self.regressions),
            "gate_checks": self.gate_checks,
            "gate_passed": self.gate_passed,
        }


class TicketStore:
    def __init__(self) -> None:
        self.tickets = {
            f"T-{number:03d}": {
                "id": f"T-{number:03d}",
                "status": "open" if number % 3 else "waiting",
                "subject": f"Support request {number}",
                "followups": [],
            }
            for number in range(101, 126)
        }
        self.side_effects: list[str] = []
        self.receipts: dict[str, tuple[str, JsonObject]] = {}

    def ticket(self, ticket_id: str) -> JsonObject | None:
        return self.tickets.get(ticket_id)


Handler = Callable[[JsonObject], ToolResult]


class ToolRegistry:
    def __init__(self, store: TicketStore) -> None:
        self.store = store
        self.specs: dict[str, ToolSpec] = {}
        self.handlers: dict[str, Handler] = {}

    def register(self, spec: ToolSpec, handler: Handler) -> None:
        if spec.name in self.specs:
            raise ValueError(f"duplicate tool: {spec.name}")
        self.specs[spec.name] = spec
        self.handlers[spec.name] = handler

    def invoke(
        self,
        call: ProposedCall,
        *,
        actor_permissions: frozenset[str],
        approved: bool,
    ) -> ToolResult:
        spec = self.specs.get(call.name)
        if spec is None:
            return _error(
                "tool_not_found",
                "routing",
                f"Unknown tool: {call.name}",
                retryable=False,
                details={"available_tools": sorted(self.specs)},
            )

        validation_errors = validate_schema(call.arguments, spec.input_schema)
        if validation_errors:
            return _error(
                "invalid_arguments",
                "validation",
                "Tool arguments did not match the input schema.",
                retryable=False,
                details={"violations": validation_errors},
            )

        if spec.required_permission not in actor_permissions:
            return _error(
                "permission_denied",
                "policy",
                f"Missing permission: {spec.required_permission}",
                retryable=False,
                details={"required_permission": spec.required_permission},
            )

        if spec.approval == "required" and not approved:
            return _error(
                "approval_required",
                "policy",
                "This write must be approved before execution.",
                retryable=False,
                details={"tool": spec.name, "effect": spec.effect},
            )

        latency_ms = call.arguments.get("simulated_latency_ms", 0)
        if isinstance(latency_ms, int) and latency_ms > spec.timeout_ms:
            return _error(
                "tool_timeout",
                "dependency",
                f"Tool exceeded its {spec.timeout_ms} ms timeout.",
                retryable=spec.retry_policy == "transient",
                details={"timeout_ms": spec.timeout_ms, "observed_ms": latency_ms},
            )

        if spec.idempotency_key:
            key = call.arguments[spec.idempotency_key]
            fingerprint = _call_fingerprint(call.arguments)
            if key in self.store.receipts:
                stored_fingerprint, stored_output = self.store.receipts[key]
                if fingerprint != stored_fingerprint:
                    return _error(
                        "idempotency_conflict",
                        "conflict",
                        "The action_id was already used with different arguments.",
                        retryable=False,
                        details={"idempotency_key": key},
                    )
                return ToolResult(
                    ok=True,
                    output=stored_output,
                    replayed=True,
                )

        try:
            result = self.handlers[call.name](call.arguments)
        except Exception as exc:  # Handler failures must not escape the tool boundary.
            return _error(
                "handler_exception",
                "internal",
                "The tool handler failed unexpectedly.",
                retryable=False,
                details={"exception_type": type(exc).__name__},
            )
        if result.ok and result.output is not None:
            output_errors = validate_schema(result.output, spec.output_schema)
            if output_errors:
                return _error(
                    "invalid_tool_output",
                    "internal",
                    "Tool output did not match its declared schema.",
                    retryable=False,
                    details={"violations": output_errors},
                )
            if spec.idempotency_key:
                key = call.arguments[spec.idempotency_key]
                self.store.receipts[key] = (
                    _call_fingerprint(call.arguments),
                    result.output,
                )
        return result

    def model_catalog(self) -> list[JsonObject]:
        return [self.specs[name].model_schema() for name in sorted(self.specs)]


class WideToolRuntime:
    """A deliberately thin control with one loosely typed operation surface."""

    MODEL_CATALOG = [
        {
            "type": "function",
            "name": "ticket_operation",
            "description": "Read or change support tickets and run ticket queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": ["operation"],
            },
        }
    ]

    def __init__(self, store: TicketStore) -> None:
        self.store = store

    def invoke(self, call: ProposedCall) -> ToolResult:
        if call.name != "ticket_operation":
            return _generic_error("tool failed")
        operation = call.arguments.get("operation")
        payload = call.arguments.get("payload", {})
        if operation == "get":
            ticket = self.store.ticket(payload.get("ticket_id", ""))
            return ToolResult(ok=True, output={"ticket": ticket})
        if operation == "record":
            ticket_id = payload.get("ticket_id", "")
            note = payload.get("note", "")
            ticket = self.store.ticket(ticket_id)
            if ticket is not None:
                ticket["followups"].append(note)
            effect = f"record:{ticket_id}:{note}"
            self.store.side_effects.append(effect)
            return ToolResult(
                ok=True,
                output={"status": "done", "ticket_id": ticket_id},
                side_effects=(effect,),
            )
        if operation == "list":
            return ToolResult(
                ok=True,
                output={"items": list(self.store.tickets.values())},
            )
        if operation == "slow":
            return _generic_error("request failed")
        return _generic_error("tool failed")


def build_candidate_registry(store: TicketStore) -> ToolRegistry:
    registry = ToolRegistry(store)

    registry.register(
        ToolSpec(
            name="get_ticket",
            description=(
                "Read one support ticket by ID. Use for inspection only; this "
                "tool never changes the ticket."
            ),
            input_schema=_object_schema(
                {"ticket_id": _string("Ticket ID such as T-102", min_length=5)},
                required=["ticket_id"],
            ),
            output_schema=_object_schema(
                {"ticket": {"type": "object"}}, required=["ticket"]
            ),
            required_permission="ticket:read",
        ),
        lambda args: _get_ticket(store, args),
    )
    registry.register(
        ToolSpec(
            name="preview_ticket_followup",
            description=(
                "Preview the exact follow-up that would be appended to a ticket. "
                "Use before record_ticket_followup when the user asks to review "
                "a change. This tool performs no write."
            ),
            input_schema=_object_schema(
                {
                    "ticket_id": _string("Ticket ID such as T-102", min_length=5),
                    "note": _string("Follow-up text to preview", min_length=1),
                },
                required=["ticket_id", "note"],
            ),
            output_schema=_object_schema(
                {
                    "ticket_id": {"type": "string"},
                    "would_append": {"type": "string"},
                    "side_effects": {"type": "integer"},
                },
                required=["ticket_id", "would_append", "side_effects"],
            ),
            required_permission="ticket:read",
        ),
        lambda args: _preview_followup(store, args),
    )
    registry.register(
        ToolSpec(
            name="record_ticket_followup",
            description=(
                "Append one approved follow-up to a support ticket. This tool "
                "writes data, requires ticket:write, and deduplicates repeated "
                "calls by action_id. Do not use it for previews."
            ),
            input_schema=_object_schema(
                {
                    "ticket_id": _string("Ticket ID such as T-102", min_length=5),
                    "note": _string("Non-empty follow-up text", min_length=1),
                    "action_id": _string(
                        "Stable idempotency key for this intended write",
                        min_length=6,
                    ),
                },
                required=["ticket_id", "note", "action_id"],
            ),
            output_schema=_object_schema(
                {
                    "receipt_id": {"type": "string"},
                    "ticket_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["recorded"]},
                },
                required=["receipt_id", "ticket_id", "status"],
            ),
            required_permission="ticket:write",
            effect="write",
            approval="required",
            idempotency_key="action_id",
        ),
        lambda args: _record_followup(store, args),
    )
    registry.register(
        ToolSpec(
            name="list_tickets",
            description=(
                "List a bounded page of support tickets. Use cursor from the "
                "previous response to continue; limit is between 1 and 5."
            ),
            input_schema=_object_schema(
                {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Maximum tickets returned in this page.",
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Opaque next cursor from a previous page.",
                    },
                },
                required=["limit"],
            ),
            output_schema=_object_schema(
                {
                    "items": {"type": "array", "maxItems": 5},
                    "next_cursor": {"type": ["string", "null"]},
                },
                required=["items", "next_cursor"],
            ),
            required_permission="ticket:read",
        ),
        lambda args: _list_tickets(store, args),
    )
    registry.register(
        ToolSpec(
            name="slow_ticket_lookup",
            description=(
                "Read a ticket through a slow dependency. On timeout, return a "
                "retryable dependency error instead of hiding the failure."
            ),
            input_schema=_object_schema(
                {
                    "ticket_id": _string("Ticket ID such as T-102", min_length=5),
                    "simulated_latency_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Deterministic latency fixture for this lab.",
                    },
                },
                required=["ticket_id", "simulated_latency_ms"],
            ),
            output_schema=_object_schema(
                {"ticket": {"type": "object"}}, required=["ticket"]
            ),
            required_permission="ticket:read",
            retry_policy="transient",
            timeout_ms=500,
        ),
        lambda args: _get_ticket(store, args),
    )
    return registry


def load_tool_cases(path: Path) -> list[ToolCase]:
    cases: list[ToolCase] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            cases.append(ToolCase.from_dict(json.loads(raw_line)))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid tool case at line {line_number}: {exc}") from exc
    return cases


def run_tool_eval(cases_path: Path) -> ToolEvalResult:
    cases = load_tool_cases(cases_path)
    runs: list[ToolRun] = []
    for case in cases:
        runs.append(_run_baseline_case(case))
        runs.append(_run_candidate_case(case))

    baseline_runs = [run for run in runs if run.strategy == "wide-tool-v1"]
    candidate_runs = [run for run in runs if run.strategy == "typed-registry-v2"]
    baseline = _summarize(
        "wide-tool-v1", baseline_runs, WideToolRuntime.MODEL_CATALOG
    )
    candidate_catalog = build_candidate_registry(TicketStore()).model_catalog()
    candidate = _summarize("typed-registry-v2", candidate_runs, candidate_catalog)
    improvements = tuple(
        case.case_id
        for case in cases
        if not _run_for(baseline_runs, case.case_id).passed
        and _run_for(candidate_runs, case.case_id).passed
    )
    regressions = tuple(
        case.case_id
        for case in cases
        if _run_for(baseline_runs, case.case_id).passed
        and not _run_for(candidate_runs, case.case_id).passed
    )
    gate_checks = {
        "candidate_passes_all_cases": candidate.passed_cases == candidate.total_cases,
        "candidate_blocks_unsafe_side_effects": candidate.unsafe_side_effects == 0,
        "candidate_deduplicates_writes": candidate.duplicate_side_effects == 0,
        "candidate_errors_are_actionable": candidate.structured_error_rate == 1.0,
        "no_case_regressions": not regressions,
    }
    return ToolEvalResult(
        version="0.5.0",
        baseline=baseline,
        candidate=candidate,
        runs=tuple(runs),
        improvements=improvements,
        regressions=regressions,
        gate_checks=gate_checks,
    )


def validate_schema(value: Any, schema: JsonObject, path: str = "$") -> list[str]:
    errors: list[str] = []
    allowed_types = schema.get("type")
    if allowed_types is not None:
        type_names = allowed_types if isinstance(allowed_types, list) else [allowed_types]
        if not any(_matches_type(value, type_name) for type_name in type_names):
            return [f"{path}: expected {' or '.join(type_names)}"]

    if value is None:
        return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in enum {schema['enum']}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errors.append(f"{path}: string is shorter than {schema['minLength']}")
    if isinstance(value, int) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value is above {schema['maximum']}")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: has more than {schema['maxItems']} items")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}.{required}: required field is missing")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key}: additional property is not allowed")
        for key, child in value.items():
            if key in properties:
                errors.extend(validate_schema(child, properties[key], f"{path}.{key}"))
    return errors


def _run_baseline_case(case: ToolCase) -> ToolRun:
    store = TicketStore()
    runtime = WideToolRuntime(store)
    results = tuple(runtime.invoke(call) for call in case.baseline_calls)
    return _grade_run("wide-tool-v1", case, results, len(store.side_effects))


def _run_candidate_case(case: ToolCase) -> ToolRun:
    store = TicketStore()
    registry = build_candidate_registry(store)
    results = tuple(
        registry.invoke(
            call,
            actor_permissions=case.actor_permissions,
            approved=case.approved,
        )
        for call in case.candidate_calls
    )
    return _grade_run("typed-registry-v2", case, results, len(store.side_effects))


def _grade_run(
    strategy: str,
    case: ToolCase,
    results: tuple[ToolResult, ...],
    side_effect_count: int,
) -> ToolRun:
    final = results[-1]
    expected = case.expected
    actual_error_code = final.error.code if final.error else None
    actual_retryable = final.error.retryable if final.error else None
    actual_output_keys = sorted(final.output) if final.output else []
    actual_output_count = (
        len(final.output.get("items", [])) if final.output else None
    )
    actual_next_cursor = (
        bool(final.output.get("next_cursor")) if final.output else False
    )
    actual_replayed = any(result.replayed for result in results)
    checks = {
        "ok": (expected.get("ok"), final.ok),
        "side_effect_count": (expected.get("side_effect_count"), side_effect_count),
    }
    optional_checks = {
        "error_code": (expected.get("error_code"), actual_error_code),
        "retryable": (expected.get("retryable"), actual_retryable),
        "replayed": (expected.get("replayed"), actual_replayed),
        "output_keys": (sorted(expected.get("output_keys", [])), actual_output_keys),
        "output_count": (expected.get("output_count"), actual_output_count),
        "has_next_cursor": (expected.get("has_next_cursor"), actual_next_cursor),
    }
    for name, pair in optional_checks.items():
        if name in expected:
            checks[name] = pair
    grades = tuple(
        ToolGrade(name=name, passed=want == got, expected=want, actual=got)
        for name, (want, got) in checks.items()
    )
    return ToolRun(
        strategy=strategy,
        case=case,
        results=results,
        grades=grades,
        side_effect_count=side_effect_count,
    )


def _summarize(
    strategy: str,
    runs: list[ToolRun],
    model_catalog: list[JsonObject],
) -> ToolSummary:
    error_runs = [run for run in runs if run.case.expected.get("ok") is False]
    actionable_errors = 0
    for run in error_runs:
        error = run.results[-1].error
        if error and error.code != "tool_failed" and error.category != "unknown":
            actionable_errors += 1
    unsafe_side_effects = sum(
        max(0, run.side_effect_count - run.case.expected["side_effect_count"])
        for run in runs
        if run.case.expected["side_effect_count"] == 0
    )
    duplicate_side_effects = sum(
        max(0, run.side_effect_count - run.case.expected["side_effect_count"])
        for run in runs
        if max(len(run.case.baseline_calls), len(run.case.candidate_calls)) > 1
    )
    schema_bytes = len(
        json.dumps(model_catalog, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return ToolSummary(
        strategy=strategy,
        total_cases=len(runs),
        passed_cases=sum(run.passed for run in runs),
        unsafe_side_effects=unsafe_side_effects,
        duplicate_side_effects=duplicate_side_effects,
        structured_error_rate=(
            actionable_errors / len(error_runs) if error_runs else 1.0
        ),
        model_schema_bytes=schema_bytes,
    )


def _run_for(runs: list[ToolRun], case_id: str) -> ToolRun:
    return next(run for run in runs if run.case.case_id == case_id)


def _get_ticket(store: TicketStore, args: JsonObject) -> ToolResult:
    ticket = store.ticket(args["ticket_id"])
    if ticket is None:
        return _error(
            "ticket_not_found",
            "not_found",
            f"Ticket {args['ticket_id']} does not exist.",
            retryable=False,
            details={"ticket_id": args["ticket_id"]},
        )
    return ToolResult(ok=True, output={"ticket": ticket})


def _record_followup(store: TicketStore, args: JsonObject) -> ToolResult:
    ticket = store.ticket(args["ticket_id"])
    if ticket is None:
        return _error(
            "ticket_not_found",
            "not_found",
            f"Ticket {args['ticket_id']} does not exist.",
            retryable=False,
            details={"ticket_id": args["ticket_id"]},
        )
    ticket["followups"].append(args["note"])
    effect = f"record:{args['ticket_id']}:{args['note']}"
    store.side_effects.append(effect)
    return ToolResult(
        ok=True,
        output={
            "receipt_id": f"receipt-{args['action_id']}",
            "ticket_id": args["ticket_id"],
            "status": "recorded",
        },
        side_effects=(effect,),
    )


def _preview_followup(store: TicketStore, args: JsonObject) -> ToolResult:
    if store.ticket(args["ticket_id"]) is None:
        return _error(
            "ticket_not_found",
            "not_found",
            f"Ticket {args['ticket_id']} does not exist.",
            retryable=False,
            details={"ticket_id": args["ticket_id"]},
        )
    return ToolResult(
        ok=True,
        output={
            "ticket_id": args["ticket_id"],
            "would_append": args["note"],
            "side_effects": 0,
        },
    )


def _list_tickets(store: TicketStore, args: JsonObject) -> ToolResult:
    try:
        start = int(args.get("cursor", "0"))
    except ValueError:
        return _error(
            "invalid_cursor",
            "validation",
            "The cursor was not issued by list_tickets.",
            retryable=False,
            details={"cursor": args.get("cursor")},
        )
    limit = args["limit"]
    tickets = list(store.tickets.values())
    page = tickets[start : start + limit]
    next_position = start + len(page)
    next_cursor = str(next_position) if next_position < len(tickets) else None
    return ToolResult(
        ok=True,
        output={"items": page, "next_cursor": next_cursor},
    )


def _object_schema(
    properties: JsonObject,
    *,
    required: list[str],
) -> JsonObject:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _string(description: str, *, min_length: int) -> JsonObject:
    return {
        "type": "string",
        "description": description,
        "minLength": min_length,
    }


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    return False


def _call_fingerprint(arguments: JsonObject) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error(
    code: str,
    category: str,
    message: str,
    *,
    retryable: bool,
    details: JsonObject,
) -> ToolResult:
    return ToolResult(
        ok=False,
        error=ToolError(
            code=code,
            category=category,
            message=message,
            retryable=retryable,
            details=details,
        ),
    )


def _generic_error(message: str) -> ToolResult:
    return _error(
        "tool_failed",
        "unknown",
        message,
        retryable=False,
        details={},
    )
