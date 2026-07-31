from __future__ import annotations

import json
from pathlib import Path

from .tools import ToolEvalResult


def write_tool_reports(
    result: ToolEvalResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tool-comparison.json"
    markdown_path = output_dir / "tool-comparison.md"
    failures_path = output_dir / "tool-failures.md"
    runs_path = output_dir / "tool-runs.jsonl"
    json_path.write_text(
        json.dumps(result.summary_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_comparison_markdown(result), encoding="utf-8")
    failures_path.write_text(_failures_markdown(result), encoding="utf-8")
    runs_path.write_text(
        "".join(
            json.dumps(run.to_dict(), ensure_ascii=False) + "\n"
            for run in result.runs
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path, failures_path, runs_path


def _comparison_markdown(result: ToolEvalResult) -> str:
    baseline = result.baseline
    candidate = result.candidate
    gate = "PASS" if result.gate_passed else "FAIL"
    lines = [
        "# Tool Contract Comparison",
        "",
        f"- Tool eval version: `{result.version}`",
        f"- Release gate: **{gate}**",
        (
            "- Scope: fixed proposed calls exercise runtime contracts; "
            "this is not a model tool-selection benchmark."
        ),
        "",
        "| Metric | wide-tool-v1 | typed-registry-v2 |",
        "| --- | ---: | ---: |",
        f"| Case pass rate | {baseline.case_pass_rate:.1%} | {candidate.case_pass_rate:.1%} |",
        f"| Unsafe side effects | {baseline.unsafe_side_effects} | {candidate.unsafe_side_effects} |",
        f"| Duplicate side effects | {baseline.duplicate_side_effects} | {candidate.duplicate_side_effects} |",
        f"| Actionable error rate | {baseline.structured_error_rate:.1%} | {candidate.structured_error_rate:.1%} |",
        f"| Model-facing schema bytes | {baseline.model_schema_bytes} | {candidate.model_schema_bytes} |",
        "",
        "The typed catalog is intentionally larger. Its safety and recovery "
        "metadata are useful, but the additional context cost is not free.",
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


def _failures_markdown(result: ToolEvalResult) -> str:
    failed_runs = [run for run in result.runs if not run.passed]
    lines = [
        "# Tool Eval Failure Ledger",
        "",
        "| Strategy | Case | Failed graders | Side effects | Final error |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for run in failed_runs:
        failed = ", ".join(
            grade.name for grade in run.grades if not grade.passed
        )
        error = run.results[-1].error
        lines.append(
            f"| {run.strategy} | {run.case.case_id} | {failed} | "
            f"{run.side_effect_count} | {error.code if error else '-'} |"
        )
    if not failed_runs:
        lines.append("| - | - | none | 0 | - |")
    lines.append("")
    return "\n".join(lines)
