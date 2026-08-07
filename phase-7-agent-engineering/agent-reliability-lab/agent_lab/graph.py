from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class GraphCompileError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    kind: str
    depends_on: tuple[str, ...]
    writes: dict[str, str]
    cost: int
    verifies: tuple[str, ...] = ()
    verdict: bool | None = None


@dataclass(frozen=True)
class GraphCase:
    case_id: str
    expected_status: str
    expected_reason: str
    budget: int
    nodes: tuple[GraphNode, ...]


@dataclass(frozen=True)
class CompiledGraph:
    case: GraphCase
    layers: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class GraphEvent:
    sequence: int
    node_id: str | None
    phase: str
    layer: int | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphCaseResult:
    case_id: str
    expected_status: str
    expected_reason: str
    status: str
    reason: str
    budget: int
    spent_budget: int
    layers: tuple[tuple[str, ...], ...]
    completed_nodes: tuple[str, ...]
    final_state: dict[str, str]
    events: tuple[GraphEvent, ...]
    matched: bool

    @property
    def merge_executed(self) -> bool:
        return "merge" in self.completed_nodes

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "expected_status": self.expected_status,
            "expected_reason": self.expected_reason,
            "status": self.status,
            "reason": self.reason,
            "budget": self.budget,
            "spent_budget": self.spent_budget,
            "layers": [list(layer) for layer in self.layers],
            "completed_nodes": list(self.completed_nodes),
            "merge_executed": self.merge_executed,
            "final_state": self.final_state,
            "events": [event.to_dict() for event in self.events],
            "matched": self.matched,
        }


@dataclass(frozen=True)
class GraphEvalResult:
    version: str
    total_cases: int
    matched_cases: int
    status_counts: dict[str, int]
    gate_checks: dict[str, bool]
    gate_passed: bool
    cases: tuple[GraphCaseResult, ...]

    def case_by_id(self, case_id: str) -> GraphCaseResult:
        return next(case for case in self.cases if case.case_id == case_id)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "comparison_scope": (
                "deterministic graph-control fixtures; not a model-quality, "
                "latency, or topology benchmark"
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


def load_graph_cases(path: Path) -> tuple[GraphCase, ...]:
    cases: list[GraphCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        data = json.loads(raw_line)
        case_id = str(data["id"])
        if case_id in seen:
            raise ValueError(f"duplicate graph case id: {case_id}")
        seen.add(case_id)
        nodes = tuple(_load_node(item) for item in data["nodes"])
        cases.append(
            GraphCase(
                case_id=case_id,
                expected_status=str(data["expected_status"]),
                expected_reason=str(data["expected_reason"]),
                budget=int(data["budget"]),
                nodes=nodes,
            )
        )
    return tuple(cases)


def compile_graph(case: GraphCase) -> CompiledGraph:
    node_by_id: dict[str, GraphNode] = {}
    for node in case.nodes:
        if node.node_id in node_by_id:
            raise GraphCompileError(
                "duplicate_node", f"duplicate node id: {node.node_id}"
            )
        node_by_id[node.node_id] = node

    for node in case.nodes:
        if node.cost <= 0:
            raise GraphCompileError(
                "invalid_node_cost",
                f"{node.node_id} cost must be a positive integer",
            )
        missing = sorted(set(node.depends_on) - set(node_by_id))
        if missing:
            raise GraphCompileError(
                "missing_dependency",
                f"{node.node_id} depends on missing node: {missing[0]}",
            )
        if node.kind == "verifier":
            if (
                not node.verifies
                or node.node_id in node.verifies
                or not set(node.verifies).issubset(node.depends_on)
                or not isinstance(node.verdict, bool)
            ):
                raise GraphCompileError(
                    "invalid_verifier",
                    f"{node.node_id} must independently verify direct dependencies",
                )
        if node.kind == "merge" and not any(
            node_by_id[dependency].kind == "verifier"
            for dependency in node.depends_on
        ):
            raise GraphCompileError(
                "unverified_merge",
                f"{node.node_id} has no verifier dependency",
            )

    remaining = set(node_by_id)
    completed: set[str] = set()
    layers: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(
            sorted(
                node_id
                for node_id in remaining
                if set(node_by_id[node_id].depends_on).issubset(completed)
            )
        )
        if not ready:
            raise GraphCompileError(
                "cycle_detected",
                f"cycle includes: {', '.join(sorted(remaining))}",
            )
        _validate_parallel_writes(ready, node_by_id)
        layers.append(ready)
        completed.update(ready)
        remaining.difference_update(ready)

    return CompiledGraph(case=case, layers=tuple(layers))


def run_graph_eval(path: Path) -> GraphEvalResult:
    cases = load_graph_cases(path)
    results = tuple(_run_case(case) for case in cases)
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    by_id = {result.case_id: result for result in results}
    valid = by_id["valid-diamond"]
    failed_verify = by_id["verifier-blocks-merge"]
    budget = by_id["budget-exhausted"]
    invalid_ids = (
        "missing-dependency",
        "cycle-detected",
        "shared-write-conflict",
    )
    gate_checks = {
        "expected_results": all(result.matched for result in results),
        "invalid_graphs_rejected": all(
            by_id[case_id].status == "invalid" for case_id in invalid_ids
        ),
        "merge_requires_verification": (
            valid.merge_executed
            and "verify" in valid.completed_nodes
            and valid.completed_nodes.index("verify")
            < valid.completed_nodes.index("merge")
            and not failed_verify.merge_executed
        ),
        "parallel_state_isolated": (
            by_id["shared-write-conflict"].reason == "shared_write_conflict"
            and all(
                key in valid.final_state
                for key in ("docs_findings", "code_findings", "policy_findings")
            )
        ),
        "independent_verifier": _has_independent_verifier(
            next(case for case in cases if case.case_id == "valid-diamond")
        ),
        "budget_honored": (
            budget.reason == "budget_exhausted"
            and budget.spent_budget <= budget.budget
            and not budget.merge_executed
        ),
        "trace_complete": all(_trace_is_complete(result) for result in results),
    }
    return GraphEvalResult(
        version="graph-control-v1",
        total_cases=len(results),
        matched_cases=sum(result.matched for result in results),
        status_counts=status_counts,
        gate_checks=gate_checks,
        gate_passed=all(gate_checks.values()),
        cases=results,
    )


def _run_case(case: GraphCase) -> GraphCaseResult:
    try:
        compiled = compile_graph(case)
    except GraphCompileError as exc:
        return _case_result(
            case,
            status="invalid",
            reason=exc.reason,
            spent=0,
            layers=(),
            completed=(),
            state={},
            events=(GraphEvent(1, None, "compile_rejected", None, exc.detail),),
        )

    node_by_id = {node.node_id: node for node in case.nodes}
    completed: list[str] = []
    state: dict[str, str] = {}
    events: list[GraphEvent] = []
    spent = 0

    for layer_index, layer in enumerate(compiled.layers):
        for node_id in layer:
            node = node_by_id[node_id]
            if spent + node.cost > case.budget:
                events.append(
                    GraphEvent(
                        len(events) + 1,
                        node_id,
                        "blocked",
                        layer_index,
                        "budget_exhausted",
                    )
                )
                return _case_result(
                    case,
                    status="blocked",
                    reason="budget_exhausted",
                    spent=spent,
                    layers=compiled.layers,
                    completed=tuple(completed),
                    state=state,
                    events=tuple(events),
                )

            events.append(
                GraphEvent(
                    len(events) + 1,
                    node_id,
                    "started",
                    layer_index,
                    node.kind,
                )
            )
            spent += node.cost
            if node.kind == "verifier" and node.verdict is False:
                events.append(
                    GraphEvent(
                        len(events) + 1,
                        node_id,
                        "failed",
                        layer_index,
                        "verifier_failed",
                    )
                )
                return _case_result(
                    case,
                    status="blocked",
                    reason="verifier_failed",
                    spent=spent,
                    layers=compiled.layers,
                    completed=tuple(completed),
                    state=state,
                    events=tuple(events),
                )

            state.update(node.writes)
            completed.append(node_id)
            events.append(
                GraphEvent(
                    len(events) + 1,
                    node_id,
                    "completed",
                    layer_index,
                    ",".join(sorted(node.writes)) or "no_state_write",
                )
            )

    return _case_result(
        case,
        status="completed",
        reason="completed",
        spent=spent,
        layers=compiled.layers,
        completed=tuple(completed),
        state=state,
        events=tuple(events),
    )


def _case_result(
    case: GraphCase,
    *,
    status: str,
    reason: str,
    spent: int,
    layers: tuple[tuple[str, ...], ...],
    completed: tuple[str, ...],
    state: dict[str, str],
    events: tuple[GraphEvent, ...],
) -> GraphCaseResult:
    return GraphCaseResult(
        case_id=case.case_id,
        expected_status=case.expected_status,
        expected_reason=case.expected_reason,
        status=status,
        reason=reason,
        budget=case.budget,
        spent_budget=spent,
        layers=layers,
        completed_nodes=completed,
        final_state=dict(state),
        events=events,
        matched=(status == case.expected_status and reason == case.expected_reason),
    )


def _load_node(data: dict[str, Any]) -> GraphNode:
    return GraphNode(
        node_id=str(data["id"]),
        kind=str(data["kind"]),
        depends_on=tuple(str(value) for value in data.get("depends_on", [])),
        writes={str(key): str(value) for key, value in data.get("writes", {}).items()},
        cost=int(data.get("cost", 1)),
        verifies=tuple(str(value) for value in data.get("verifies", [])),
        verdict=data.get("verdict"),
    )


def _validate_parallel_writes(
    layer: tuple[str, ...], node_by_id: dict[str, GraphNode]
) -> None:
    owner_by_key: dict[str, str] = {}
    for node_id in layer:
        for key in node_by_id[node_id].writes:
            if key in owner_by_key:
                raise GraphCompileError(
                    "shared_write_conflict",
                    f"{owner_by_key[key]} and {node_id} both write {key}",
                )
            owner_by_key[key] = node_id


def _has_independent_verifier(case: GraphCase) -> bool:
    verifiers = [node for node in case.nodes if node.kind == "verifier"]
    return bool(verifiers) and all(
        node.node_id not in node.verifies
        and set(node.verifies).issubset(node.depends_on)
        for node in verifiers
    )


def _trace_is_complete(result: GraphCaseResult) -> bool:
    if result.status == "invalid":
        return any(event.phase == "compile_rejected" for event in result.events)
    phases_by_node: dict[str, set[str]] = {}
    for event in result.events:
        if event.node_id is not None:
            phases_by_node.setdefault(event.node_id, set()).add(event.phase)
    for node_id in result.completed_nodes:
        if not {"started", "completed"}.issubset(phases_by_node.get(node_id, set())):
            return False
    terminal = result.events[-1].phase if result.events else None
    return terminal in {"completed", "failed", "blocked"}
