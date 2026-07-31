from __future__ import annotations

import json
from pathlib import Path

from .harness import HarnessEvalResult


def write_harness_reports(
    result: HarnessEvalResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "harness-comparison.json"
    markdown_path = output_dir / "harness-comparison.md"
    failures_path = output_dir / "harness-failures.md"
    runs_path = output_dir / "harness-runs.jsonl"
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


def _comparison_markdown(result: HarnessEvalResult) -> str:
    baseline = result.baseline
    candidate = result.candidate
    gate = "PASS" if result.gate_passed else "FAIL"
    lines = [
        "# Harness Boundary Comparison",
        "",
        f"- Harness eval version: `{result.version}`",
        f"- Release gate: **{gate}**",
        (
            "- Scope: deterministic boundary conformance; "
            "not model quality or SDK ranking."
        ),
        "",
        "| Metric | inline-loop-v1 | minimal-harness-v1 |",
        "| --- | ---: | ---: |",
        f"| Case pass rate | {baseline.case_pass_rate:.1%} | {candidate.case_pass_rate:.1%} |",
        f"| Explicit state rate | {baseline.explicit_state_rate:.1%} | {candidate.explicit_state_rate:.1%} |",
        f"| Sensitive-action control | {baseline.sensitive_action_control_rate:.1%} | {candidate.sensitive_action_control_rate:.1%} |",
        f"| Checkpoint before pause | {baseline.checkpoint_before_pause_rate:.1%} | {candidate.checkpoint_before_pause_rate:.1%} |",
        f"| Trace completeness | {baseline.trace_completeness_rate:.1%} | {candidate.trace_completeness_rate:.1%} |",
        f"| Duplicate side effects | {baseline.duplicate_side_effects} | {candidate.duplicate_side_effects} |",
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


def _failures_markdown(result: HarnessEvalResult) -> str:
    failed_runs = [run for run in result.runs if not run.passed]
    lines = [
        "# Harness Eval Failure Ledger",
        "",
        "| Strategy | Case | Status | Failed graders | Side effects |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for run in failed_runs:
        failed = ", ".join(
            grade.name for grade in run.grades if not grade.passed
        )
        lines.append(
            f"| {run.strategy} | {run.case.case_id} | {run.state.status} | "
            f"{failed} | {run.side_effect_count} |"
        )
    if not failed_runs:
        lines.append("| - | - | - | none | 0 |")
    lines.append("")
    return "\n".join(lines)
