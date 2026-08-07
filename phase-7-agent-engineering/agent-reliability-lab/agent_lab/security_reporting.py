from __future__ import annotations

import json
from pathlib import Path

from .security import SecurityEvalResult


def write_security_reports(
    result: SecurityEvalResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "security-review.json"
    markdown_path = output_dir / "security-review.md"
    runs_path = output_dir / "security-runs.jsonl"
    failures_path = output_dir / "security-failures.md"

    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_review_markdown(result), encoding="utf-8")
    runs_path.write_text(_runs_jsonl(result), encoding="utf-8")
    failures_path.write_text(_failures_markdown(result), encoding="utf-8")
    return json_path, markdown_path, runs_path, failures_path


def _review_markdown(result: SecurityEvalResult) -> str:
    lines = [
        "# Agent Human Control Review",
        "",
        f"- Version: `{result.version}`",
        f"- Review gate: **{'PASS' if result.gate_passed else 'FAIL'}**",
        f"- Matched outcomes: `{result.matched_cases}/{result.total_cases}`",
        "- Scope: deterministic approval-control fixtures, not a penetration test",
        "",
        "## Release gate",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for check, passed in result.gate_checks.items():
        lines.append(f"| `{check}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Case outcomes",
            "",
            "| Case | Risk | Expected | Actual | Reason | Effects |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for case in result.cases:
        lines.append(
            f"| `{case.case_id}` | `{case.risk}` | `{case.expected_status}` | "
            f"`{case.status}` | `{case.reason}` | {case.mutation_count} |"
        )
    lines.extend(
        [
            "",
            "PASS means the fixtures matched the declared approval contract. "
            "It does not prove that a model, IAM setup, or production system is secure.",
            "",
        ]
    )
    return "\n".join(lines)


def _runs_jsonl(result: SecurityEvalResult) -> str:
    return "\n".join(
        json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True)
        for case in result.cases
    ) + "\n"


def _failures_markdown(result: SecurityEvalResult) -> str:
    stopped = [case for case in result.cases if case.status != "completed"]
    lines = [
        "# Human Control Stop Review",
        "",
        "These stopped paths are expected policy outcomes, not hidden successes.",
        "",
        "| Case | Status | Reason | Effects |",
        "| --- | --- | --- | ---: |",
    ]
    for case in stopped:
        lines.append(
            f"| `{case.case_id}` | `{case.status}` | `{case.reason}` | "
            f"{case.mutation_count} |"
        )
    lines.append("")
    return "\n".join(lines)

