from __future__ import annotations

import json
from pathlib import Path

from .context import ContextEvalResult


def write_context_reports(
    result: ContextEvalResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "context-comparison.json"
    markdown_path = output_dir / "context-comparison.md"
    failures_path = output_dir / "context-failures.md"
    packets_path = output_dir / "context-packets.jsonl"
    json_path.write_text(
        json.dumps(result.summary_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_comparison_markdown(result), encoding="utf-8")
    failures_path.write_text(_failures_markdown(result), encoding="utf-8")
    packets_path.write_text(
        "".join(
            json.dumps(run.to_dict(), ensure_ascii=False) + "\n"
            for run in result.runs
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path, failures_path, packets_path


def _comparison_markdown(result: ContextEvalResult) -> str:
    baseline = result.baseline
    candidate = result.candidate
    gate = "PASS" if result.gate_passed else "FAIL"
    lines = [
        "# Context Architecture Comparison",
        "",
        f"- Context eval version: `{result.version}`",
        f"- Release gate: **{gate}**",
        "",
        "| Metric | dump-all-v1 | context-packet-v1 |",
        "| --- | ---: | ---: |",
        f"| Case pass rate | {baseline.case_pass_rate:.1%} | {candidate.case_pass_rate:.1%} |",
        f"| Required topic coverage | {baseline.required_topic_coverage:.1%} | {candidate.required_topic_coverage:.1%} |",
        f"| Invalid-source cases | {baseline.invalid_source_cases} | {candidate.invalid_source_cases} |",
        f"| Irrelevant-source cases | {baseline.irrelevant_source_cases} | {candidate.irrelevant_source_cases} |",
        f"| Budget compliance | {baseline.budget_compliance_rate:.1%} | {candidate.budget_compliance_rate:.1%} |",
        f"| Missing-evidence accuracy | {baseline.missing_evidence_accuracy:.1%} | {candidate.missing_evidence_accuracy:.1%} |",
        f"| Average estimated tokens | {baseline.average_estimated_tokens:.2f} | {candidate.average_estimated_tokens:.2f} |",
        "",
        "Estimated tokens come from the lab's deterministic estimator, not a provider tokenizer.",
        "",
        "## Changed cases",
        "",
        f"- Improvements: {', '.join(result.improvements) or 'none'}",
        f"- Regressions: {', '.join(result.regressions) or 'none'}",
        "",
        "## Release gate",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in result.gate_checks.items()
    )
    lines.append("")
    return "\n".join(lines)


def _failures_markdown(result: ContextEvalResult) -> str:
    failed_runs = [run for run in result.runs if not run.passed]
    lines = [
        "# Context Eval Failure Ledger",
        "",
        "| Strategy | Case | Failed graders | Selected sources | Missing topics |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in failed_runs:
        failed_grades = ", ".join(
            grade.name for grade in run.grades if not grade.passed
        )
        selected = ", ".join(
            source.source_id for source in run.packet.selected
        ) or "-"
        missing = ", ".join(run.packet.missing_topics) or "-"
        lines.append(
            f"| {run.strategy} | {run.case.case_id} | {failed_grades} | "
            f"{selected} | {missing} |"
        )
    if not failed_runs:
        lines.append("| - | - | none | - | - |")
    lines.append("")
    return "\n".join(lines)
