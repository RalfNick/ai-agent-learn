from __future__ import annotations

import json
from pathlib import Path

from .durable import DurableEvalResult


def write_durable_reports(
    result: DurableEvalResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "durable-comparison.json"
    markdown_path = output_dir / "durable-comparison.md"
    failures_path = output_dir / "durable-failures.md"
    runs_path = output_dir / "durable-runs.jsonl"
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


def _comparison_markdown(result: DurableEvalResult) -> str:
    baseline = result.baseline
    candidate = result.candidate
    gate = "PASS" if result.gate_passed else "FAIL"
    lines = [
        "# Durable Loop Fault Comparison",
        "",
        f"- Durable eval version: `{result.version}`",
        f"- Release gate: **{gate}**",
        (
            "- Scope: deterministic process-loss and dependency-fault fixtures; "
            "not a distributed runtime benchmark."
        ),
        "",
        "| Metric | process-loop-v1 | durable-loop-v1 |",
        "| --- | ---: | ---: |",
        f"| Case pass rate | {baseline.case_pass_rate:.1%} | {candidate.case_pass_rate:.1%} |",
        f"| Total model attempts | {baseline.total_model_attempts} | {candidate.total_model_attempts} |",
        f"| Duplicate side effects | {baseline.duplicate_side_effects} | {candidate.duplicate_side_effects} |",
        f"| Blind retries | {baseline.blind_retries} | {candidate.blind_retries} |",
        f"| Explicit terminal rate | {baseline.explicit_terminal_rate:.1%} | {candidate.explicit_terminal_rate:.1%} |",
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


def _failures_markdown(result: DurableEvalResult) -> str:
    failed_runs = [run for run in result.runs if not run.passed]
    lines = [
        "# Durable Loop Failure Ledger",
        "",
        "| Strategy | Case | Failed graders | Effects | Duplicates | Final state |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for run in failed_runs:
        failed = ", ".join(
            grade.name for grade in run.grades if not grade.passed
        )
        lines.append(
            f"| {run.strategy} | {run.case.case_id} | {failed} | "
            f"{run.side_effect_count} | {run.duplicate_side_effects} | "
            f"{run.state.status} |"
        )
    if not failed_runs:
        lines.append("| - | - | none | 0 | 0 | - |")
    lines.append("")
    return "\n".join(lines)
