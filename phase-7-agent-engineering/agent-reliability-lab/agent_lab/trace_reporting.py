from __future__ import annotations

import json
from pathlib import Path

from .tracing import TraceReviewResult


def write_trace_reports(
    result: TraceReviewResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "trace-review.json"
    markdown_path = output_dir / "trace-review.md"
    failures_path = output_dir / "trace-failures.md"
    traces_path = output_dir / "traces.jsonl"

    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_review_markdown(result), encoding="utf-8")
    failures_path.write_text(_failures_markdown(result), encoding="utf-8")
    traces_path.write_text(_traces_jsonl(result), encoding="utf-8")
    return json_path, markdown_path, failures_path, traces_path


def _review_markdown(result: TraceReviewResult) -> str:
    gate = "PASS" if result.gate_passed else "FAIL"
    lines = [
        "# Agent Trace Review",
        "",
        f"- Trace checkpoint: `{result.version}`",
        f"- Review gate: **{gate}**",
        (
            "- Scope: deterministic trace-contract fixtures; not a production "
            "observability benchmark."
        ),
        "",
        "## Comparison",
        "",
        "| Evidence source | Debugging question answer rate |",
        "| --- | ---: |",
        f"| one-line log | {result.baseline_question_answer_rate:.1%} |",
        f"| structured trace | {result.candidate_question_answer_rate:.1%} |",
        "",
        "## Five debugging questions",
        "",
        "| Case | Context sources | Tool path | First failure | Retry evidence | Versions |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in result.cases:
        answers = {question.question: question for question in case.questions}
        cells = [
            "yes" if answers[name].answered else "no"
            for name in (
                "context_sources",
                "tool_path",
                "first_failure",
                "retry_evidence",
                "version_tuple",
            )
        ]
        lines.append(f"| {case.case.case_id} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Review gate", ""])
    lines.extend(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in result.gate_checks.items()
    )
    lines.append("")
    return "\n".join(lines)


def _failures_markdown(result: TraceReviewResult) -> str:
    lines = [
        "# Trace Finding Ledger",
        "",
        "These fixtures are intentionally mixed: a finding is correct when it matches the case contract.",
        "",
        "| Case | Finding | Span | Detail | Expected match |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in result.cases:
        if not case.findings:
            lines.append(f"| {case.case.case_id} | none | - | clean trace | yes |")
            continue
        for finding in case.findings:
            detail = finding.detail.replace("|", "\\|")
            lines.append(
                f"| {case.case.case_id} | {finding.code} | "
                f"{finding.span_id or '-'} | {detail} | "
                f"{'yes' if case.passed else 'no'} |"
            )
    lines.append("")
    return "\n".join(lines)


def _traces_jsonl(result: TraceReviewResult) -> str:
    rows = []
    for case in result.cases:
        for span in case.spans:
            rows.append(
                json.dumps(
                    {"case_id": case.case.case_id, **span.to_dict()},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    return "\n".join(rows) + "\n"
