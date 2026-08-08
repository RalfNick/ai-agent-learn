from __future__ import annotations

import json
from pathlib import Path

from .operations import OperationsEvalResult


def write_operations_reports(
    result: OperationsEvalResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "operations-review.json"
    markdown_path = output_dir / "operations-review.md"
    runs_path = output_dir / "operations-runs.jsonl"
    evals_path = output_dir / "incident-evals.jsonl"
    failures_path = output_dir / "operations-failures.md"

    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_review_markdown(result), encoding="utf-8")
    runs_path.write_text(_runs_jsonl(result), encoding="utf-8")
    evals_path.write_text(_evals_jsonl(result), encoding="utf-8")
    failures_path.write_text(_failures_markdown(result), encoding="utf-8")
    return json_path, markdown_path, runs_path, evals_path, failures_path


def _review_markdown(result: OperationsEvalResult) -> str:
    lines = [
        "# Agent Production Loop Review",
        "",
        f"- Version: `{result.version}`",
        f"- Release gate: **{'PASS' if result.gate_passed else 'FAIL'}**",
        f"- Matched decisions: `{result.matched_cases}/{result.total_cases}`",
        f"- Regression candidates: `{len(result.eval_candidates)}`",
        "- Scope: deterministic policy fixtures, not a production SRE audit",
        "- Decision boundary: emits policy decisions; does not execute rollback or compensation",
        "- Privacy boundary: uses pre-sanitized metadata fixtures; no redaction pipeline is implemented",
        "",
        "## Release checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for check, passed in result.gate_checks.items():
        lines.append(f"| `{check}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Window decisions",
            "",
            "| Window | Expected | Actual | Reason | Eval candidate |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in result.cases:
        lines.append(
            f"| `{case.case_id}` | `{case.expected_action}` | "
            f"`{case.decision.action}` | `{case.decision.reason}` | "
            f"{'yes' if case.eval_candidate else 'no'} |"
        )
    lines.extend(
        [
            "",
            "PASS means the fixed windows matched the declared operations contract. "
            "It does not prove capacity, model quality, security, or a universal SLO.",
            "",
        ]
    )
    return "\n".join(lines)


def _runs_jsonl(result: OperationsEvalResult) -> str:
    return "\n".join(
        json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True)
        for case in result.cases
    ) + "\n"


def _evals_jsonl(result: OperationsEvalResult) -> str:
    return "\n".join(
        json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True)
        for candidate in result.eval_candidates
    ) + "\n"


def _failures_markdown(result: OperationsEvalResult) -> str:
    mismatches = [case for case in result.cases if not case.matched]
    lines = [
        "# Production Loop Failure Review",
        "",
        "## Contract mismatches",
        "",
    ]
    if not mismatches:
        lines.append("No declared outcome mismatches.")
    else:
        lines.extend(
            [
                "| Window | Expected | Actual | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for case in mismatches:
            lines.append(
                f"| `{case.case_id}` | `{case.expected_action}` | "
                f"`{case.decision.action}` | `{case.decision.reason}` |"
            )
    lines.extend(["", "## Controlled incidents", ""])
    incidents = [case for case in result.cases if case.decision.incident]
    lines.extend(
        [
            "| Window | Action | Reason | Eval task |",
            "| --- | --- | --- | --- |",
        ]
    )
    for case in incidents:
        task_id = case.eval_candidate.task_id if case.eval_candidate else "missing"
        lines.append(
            f"| `{case.case_id}` | `{case.decision.action}` | "
            f"`{case.decision.reason}` | `{task_id}` |"
        )
    lines.append("")
    return "\n".join(lines)
