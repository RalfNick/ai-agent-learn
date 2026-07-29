from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContextSource:
    source_id: str
    kind: str
    title: str
    content: str
    locator: str
    trust: str
    authority: int
    updated_at: date
    valid_until: date | None
    topics: list[str]
    sensitivity: int
    canonical_key: str

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(f"{self.title}\n{self.content}\n{self.locator}")

    def to_packet_dict(self) -> dict[str, Any]:
        return {
            "id": self.source_id,
            "kind": self.kind,
            "title": self.title,
            "content": self.content,
            "locator": self.locator,
            "trust": self.trust,
            "updated_at": self.updated_at.isoformat(),
            "valid_until": (
                self.valid_until.isoformat() if self.valid_until is not None else None
            ),
            "topics": self.topics,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass(frozen=True)
class ContextCase:
    case_id: str
    question: str
    as_of: date
    budget: int
    clearance: int
    required_topics: list[str]
    expected_missing_topics: list[str]
    forbidden_source_ids: list[str]
    candidate_source_ids: list[str]


@dataclass(frozen=True)
class ExcludedSource:
    source_id: str
    reason: str
    locator: str


@dataclass(frozen=True)
class ContextPacket:
    schema_version: str
    strategy: str
    case_id: str
    question: str
    as_of: str
    selected: list[ContextSource]
    excluded: list[ExcludedSource]
    missing_topics: list[str]
    budget_limit: int
    budget_used: int
    fingerprint: str
    rendered_context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "case_id": self.case_id,
            "question": self.question,
            "as_of": self.as_of,
            "selected": [source.to_packet_dict() for source in self.selected],
            "excluded": [asdict(source) for source in self.excluded],
            "missing_topics": self.missing_topics,
            "budget": {
                "limit_estimated_tokens": self.budget_limit,
                "used_estimated_tokens": self.budget_used,
                "remaining_estimated_tokens": self.budget_limit - self.budget_used,
            },
            "fingerprint": self.fingerprint,
            "rendered_context": self.rendered_context,
        }


@dataclass(frozen=True)
class ContextGrade:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ContextRun:
    strategy: str
    case: ContextCase
    packet: ContextPacket
    grades: list[ContextGrade]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "case": {
                "id": self.case.case_id,
                "required_topics": self.case.required_topics,
                "expected_missing_topics": self.case.expected_missing_topics,
                "forbidden_source_ids": self.case.forbidden_source_ids,
            },
            "packet": self.packet.to_dict(),
            "grades": [asdict(grade) for grade in self.grades],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ContextSummary:
    strategy: str
    cases: int
    passed_cases: int
    case_pass_rate: float
    available_required_topics: int
    covered_required_topics: int
    required_topic_coverage: float
    invalid_source_cases: int
    irrelevant_source_cases: int
    budget_compliance_rate: float
    missing_evidence_accuracy: float
    average_estimated_tokens: float


@dataclass(frozen=True)
class ContextEvalResult:
    version: str
    baseline: ContextSummary
    candidate: ContextSummary
    improvements: list[str]
    regressions: list[str]
    gate_checks: dict[str, bool]
    gate_passed: bool
    runs: list[ContextRun]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "baseline": asdict(self.baseline),
            "candidate": asdict(self.candidate),
            "improvements": self.improvements,
            "regressions": self.regressions,
            "gate_checks": self.gate_checks,
            "gate_passed": self.gate_passed,
        }


def estimate_tokens(text: str) -> int:
    """Return a deterministic estimate, not a provider tokenizer count."""
    cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    non_cjk = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff]", " ", text)
    pieces = re.findall(r"[A-Za-z0-9_./:-]+|[^\sA-Za-z0-9_./:-]", non_cjk)
    return max(1, cjk + len(pieces))


def load_context_sources(path: Path) -> list[ContextSource]:
    sources: list[ContextSource] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        data = json.loads(raw_line)
        source_id = str(data["id"])
        if source_id in seen:
            raise ValueError(f"duplicate context source id on line {line_number}: {source_id}")
        seen.add(source_id)
        trust = str(data["trust"])
        if trust not in {"trusted", "untrusted"}:
            raise ValueError(f"invalid trust on line {line_number}: {trust}")
        sources.append(
            ContextSource(
                source_id=source_id,
                kind=str(data["kind"]),
                title=str(data["title"]),
                content=str(data["content"]),
                locator=str(data["locator"]),
                trust=trust,
                authority=int(data["authority"]),
                updated_at=date.fromisoformat(str(data["updated_at"])),
                valid_until=(
                    date.fromisoformat(str(data["valid_until"]))
                    if data.get("valid_until")
                    else None
                ),
                topics=[str(topic) for topic in data["topics"]],
                sensitivity=int(data["sensitivity"]),
                canonical_key=str(data.get("canonical_key", source_id)),
            )
        )
    if not sources:
        raise ValueError("context source dataset must not be empty")
    return sources


def load_context_cases(path: Path) -> list[ContextCase]:
    cases: list[ContextCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        data = json.loads(raw_line)
        case_id = str(data["id"])
        if case_id in seen:
            raise ValueError(f"duplicate context case id on line {line_number}: {case_id}")
        seen.add(case_id)
        cases.append(
            ContextCase(
                case_id=case_id,
                question=str(data["question"]),
                as_of=date.fromisoformat(str(data["as_of"])),
                budget=int(data["budget"]),
                clearance=int(data["clearance"]),
                required_topics=[str(topic) for topic in data["required_topics"]],
                expected_missing_topics=[
                    str(topic) for topic in data.get("expected_missing_topics", [])
                ],
                forbidden_source_ids=[
                    str(source_id) for source_id in data.get("forbidden_source_ids", [])
                ],
                candidate_source_ids=[
                    str(source_id) for source_id in data["candidate_source_ids"]
                ],
            )
        )
    if not cases:
        raise ValueError("context case dataset must not be empty")
    return cases


def run_context_eval(
    cases_path: Path,
    sources_path: Path,
    *,
    budget_override: int | None = None,
) -> ContextEvalResult:
    cases = load_context_cases(cases_path)
    sources = load_context_sources(sources_path)
    source_map = {source.source_id: source for source in sources}
    runs = []
    for strategy in ("dump-all-v1", "context-packet-v1"):
        for case in cases:
            packet = assemble_context(
                case,
                source_map,
                strategy=strategy,
                budget_override=budget_override,
            )
            grades = grade_context(case, packet)
            runs.append(
                ContextRun(
                    strategy=strategy,
                    case=case,
                    packet=packet,
                    grades=grades,
                    passed=all(grade.passed for grade in grades),
                )
            )

    baseline_runs = [run for run in runs if run.strategy == "dump-all-v1"]
    candidate_runs = [run for run in runs if run.strategy == "context-packet-v1"]
    baseline = summarize_context("dump-all-v1", baseline_runs)
    candidate = summarize_context("context-packet-v1", candidate_runs)
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
    gate_checks = {
        "candidate_not_worse_overall": (
            candidate.case_pass_rate >= baseline.case_pass_rate
        ),
        "at_least_one_measured_improvement": bool(improvements),
        "no_case_regressions": not regressions,
        "all_required_available_evidence_retained": (
            candidate.required_topic_coverage == 1.0
        ),
        "no_forbidden_source_exposure": candidate.invalid_source_cases == 0,
        "no_irrelevant_source_exposure": candidate.irrelevant_source_cases == 0,
        "all_packets_within_budget": candidate.budget_compliance_rate == 1.0,
        "missing_evidence_is_explicit": candidate.missing_evidence_accuracy == 1.0,
    }
    return ContextEvalResult(
        version="0.3.0",
        baseline=baseline,
        candidate=candidate,
        improvements=improvements,
        regressions=regressions,
        gate_checks=gate_checks,
        gate_passed=all(gate_checks.values()),
        runs=runs,
    )


def assemble_context(
    case: ContextCase,
    source_map: dict[str, ContextSource],
    *,
    strategy: str,
    budget_override: int | None = None,
) -> ContextPacket:
    budget = budget_override if budget_override is not None else case.budget
    if budget < 1:
        raise ValueError("context budget must be at least 1")
    try:
        candidates = [source_map[source_id] for source_id in case.candidate_source_ids]
    except KeyError as exc:
        raise ValueError(f"context case references missing source: {exc.args[0]}") from exc

    if strategy == "dump-all-v1":
        selected, excluded = _assemble_dump_all(candidates, budget)
    elif strategy == "context-packet-v1":
        selected, excluded = _assemble_packet(case, candidates, budget)
    else:
        raise ValueError(f"unknown context strategy: {strategy}")

    selected_topics = {topic for source in selected for topic in source.topics}
    missing_topics = sorted(set(case.required_topics) - selected_topics)
    used = sum(source.estimated_tokens for source in selected)
    fingerprint_input = json.dumps(
        {
            "strategy": strategy,
            "case": case.case_id,
            "sources": [source.source_id for source in selected],
            "as_of": case.as_of.isoformat(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()[:16]
    rendered = _render_context(case, selected, missing_topics)
    return ContextPacket(
        schema_version="0.3.0",
        strategy=strategy,
        case_id=case.case_id,
        question=case.question,
        as_of=case.as_of.isoformat(),
        selected=selected,
        excluded=excluded,
        missing_topics=missing_topics,
        budget_limit=budget,
        budget_used=used,
        fingerprint=fingerprint,
        rendered_context=rendered,
    )


def grade_context(case: ContextCase, packet: ContextPacket) -> list[ContextGrade]:
    selected_ids = {source.source_id for source in packet.selected}
    selected_topics = {topic for source in packet.selected for topic in source.topics}
    expected_available = set(case.required_topics) - set(case.expected_missing_topics)
    missing_available = sorted(expected_available - selected_topics)
    forbidden_present = sorted(selected_ids & set(case.forbidden_source_ids))
    irrelevant = sorted(
        source.source_id
        for source in packet.selected
        if not set(source.topics) & set(case.required_topics)
    )
    traceability_ok = all(
        source.locator and source.updated_at for source in packet.selected
    )
    return [
        ContextGrade(
            name="required_evidence",
            passed=not missing_available,
            detail=(
                "all available required topics selected"
                if not missing_available
                else f"missing={','.join(missing_available)}"
            ),
        ),
        ContextGrade(
            name="missing_evidence",
            passed=set(packet.missing_topics) == set(case.expected_missing_topics),
            detail=(
                f"actual={packet.missing_topics}, "
                f"expected={case.expected_missing_topics}"
            ),
        ),
        ContextGrade(
            name="forbidden_sources",
            passed=not forbidden_present,
            detail=(
                "no forbidden sources selected"
                if not forbidden_present
                else f"selected={','.join(forbidden_present)}"
            ),
        ),
        ContextGrade(
            name="relevance",
            passed=not irrelevant,
            detail=(
                "all selected sources match a required topic"
                if not irrelevant
                else f"irrelevant={','.join(irrelevant)}"
            ),
        ),
        ContextGrade(
            name="budget",
            passed=packet.budget_used <= packet.budget_limit,
            detail=f"used={packet.budget_used}, limit={packet.budget_limit}",
        ),
        ContextGrade(
            name="traceability",
            passed=traceability_ok,
            detail="all selected sources retain locator and updated_at",
        ),
    ]


def summarize_context(strategy: str, runs: list[ContextRun]) -> ContextSummary:
    available_topics = 0
    covered_topics = 0
    budget_passes = 0
    missing_passes = 0
    invalid_source_cases = 0
    irrelevant_source_cases = 0
    used_tokens = []
    for run in runs:
        expected_available = set(run.case.required_topics) - set(
            run.case.expected_missing_topics
        )
        selected_topics = {
            topic for source in run.packet.selected for topic in source.topics
        }
        available_topics += len(expected_available)
        covered_topics += len(expected_available & selected_topics)
        grade_map = {grade.name: grade for grade in run.grades}
        budget_passes += grade_map["budget"].passed
        missing_passes += grade_map["missing_evidence"].passed
        invalid_source_cases += not grade_map["forbidden_sources"].passed
        irrelevant_source_cases += not grade_map["relevance"].passed
        used_tokens.append(run.packet.budget_used)
    return ContextSummary(
        strategy=strategy,
        cases=len(runs),
        passed_cases=sum(run.passed for run in runs),
        case_pass_rate=_rate(sum(run.passed for run in runs), len(runs)),
        available_required_topics=available_topics,
        covered_required_topics=covered_topics,
        required_topic_coverage=_rate(covered_topics, available_topics),
        invalid_source_cases=invalid_source_cases,
        irrelevant_source_cases=irrelevant_source_cases,
        budget_compliance_rate=_rate(budget_passes, len(runs)),
        missing_evidence_accuracy=_rate(missing_passes, len(runs)),
        average_estimated_tokens=(
            round(sum(used_tokens) / len(used_tokens), 2) if used_tokens else 0.0
        ),
    )


def _assemble_dump_all(
    candidates: list[ContextSource], budget: int
) -> tuple[list[ContextSource], list[ExcludedSource]]:
    selected = []
    excluded = []
    used = 0
    cut = False
    for source in candidates:
        if cut:
            excluded.append(_excluded(source, "after_prefix_cut"))
            continue
        if used + source.estimated_tokens > budget:
            excluded.append(_excluded(source, "budget_prefix_cut"))
            cut = True
            continue
        selected.append(source)
        used += source.estimated_tokens
    return selected, excluded


def _assemble_packet(
    case: ContextCase,
    candidates: list[ContextSource],
    budget: int,
) -> tuple[list[ContextSource], list[ExcludedSource]]:
    eligible: list[ContextSource] = []
    excluded: list[ExcludedSource] = []
    for source in candidates:
        reason = _ineligible_reason(case, source)
        if reason is None:
            eligible.append(source)
        else:
            excluded.append(_excluded(source, reason))

    deduplicated: dict[str, ContextSource] = {}
    for source in eligible:
        existing = deduplicated.get(source.canonical_key)
        if existing is None or _source_rank(source) > _source_rank(existing):
            if existing is not None:
                excluded.append(_excluded(existing, "superseded"))
            deduplicated[source.canonical_key] = source
        else:
            excluded.append(_excluded(source, "superseded"))

    remaining = list(deduplicated.values())
    selected: list[ContextSource] = []
    uncovered = set(case.required_topics)
    used = 0
    while uncovered:
        fitting = [
            source
            for source in remaining
            if set(source.topics) & uncovered
            and used + source.estimated_tokens <= budget
        ]
        if not fitting:
            break
        chosen = max(
            fitting,
            key=lambda source: (
                len(set(source.topics) & uncovered),
                source.authority,
                source.updated_at,
                -source.estimated_tokens,
                source.source_id,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        used += chosen.estimated_tokens
        uncovered -= set(chosen.topics)

    for source in remaining:
        reason = (
            "budget"
            if set(source.topics) & set(case.required_topics)
            and used + source.estimated_tokens > budget
            else "not_required"
        )
        excluded.append(_excluded(source, reason))
    return selected, excluded


def _ineligible_reason(case: ContextCase, source: ContextSource) -> str | None:
    if source.trust != "trusted":
        return "untrusted"
    if source.sensitivity > case.clearance:
        return "clearance"
    if source.valid_until is not None and source.valid_until < case.as_of:
        return "expired"
    return None


def _source_rank(source: ContextSource) -> tuple[date, int, int, str]:
    return (
        source.updated_at,
        source.authority,
        -source.estimated_tokens,
        source.source_id,
    )


def _excluded(source: ContextSource, reason: str) -> ExcludedSource:
    return ExcludedSource(
        source_id=source.source_id,
        reason=reason,
        locator=source.locator,
    )


def _render_context(
    case: ContextCase,
    selected: list[ContextSource],
    missing_topics: list[str],
) -> str:
    lines = [
        "<task>",
        case.question,
        "</task>",
        "<evidence>",
    ]
    for source in selected:
        lines.extend(
            [
                (
                    f'<source id="{source.source_id}" '
                    f'updated_at="{source.updated_at.isoformat()}" '
                    f'locator="{source.locator}">'
                ),
                source.content,
                "</source>",
            ]
        )
    lines.append("</evidence>")
    if missing_topics:
        lines.extend(
            [
                "<missing_evidence>",
                ", ".join(missing_topics),
                "</missing_evidence>",
            ]
        )
    return "\n".join(lines)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
