from __future__ import annotations

import json
from pathlib import Path

from .evals import EvalResult


def write_eval_reports(
    result: EvalResult, output_dir: Path
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    failures_path = output_dir / "failures.md"
    trials_path = output_dir / "trials.jsonl"
    json_path.write_text(
        json.dumps(result.summary_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_comparison_markdown(result), encoding="utf-8")
    failures_path.write_text(_failures_markdown(result), encoding="utf-8")
    trials_path.write_text(
        "".join(
            json.dumps(trial.to_dict(), ensure_ascii=False) + "\n"
            for trial in result.trials
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path, failures_path, trials_path


def _comparison_markdown(result: EvalResult) -> str:
    baseline = result.baseline
    candidate = result.candidate
    gate = "PASS" if result.gate_passed else "FAIL"
    lines = [
        "# Agent Eval Comparison",
        "",
        f"- Eval version: `{result.version}`",
        f"- Trials per task: `{result.trials_per_task}`",
        f"- Release gate: **{gate}**",
        "",
        "| Metric | Baseline | Candidate |",
        "| --- | ---: | ---: |",
        f"| Trial pass rate | {baseline.trial_pass_rate:.1%} | {candidate.trial_pass_rate:.1%} |",
        f"| Task pass rate | {baseline.task_pass_rate:.1%} | {candidate.task_pass_rate:.1%} |",
        f"| Correct abstention rate | {baseline.correct_abstention_rate:.1%} | {candidate.correct_abstention_rate:.1%} |",
        f"| False answer rate | {baseline.false_answer_rate:.1%} | {candidate.false_answer_rate:.1%} |",
        f"| Stability rate | {baseline.stability_rate:.1%} | {candidate.stability_rate:.1%} |",
        f"| Median latency (local ms) | {baseline.median_latency_ms:.3f} | {candidate.median_latency_ms:.3f} |",
        f"| p95 latency (local ms) | {baseline.p95_latency_ms:.3f} | {candidate.p95_latency_ms:.3f} |",
        "",
        "Local latency is diagnostic only and is not part of the release gate.",
        "",
        "## Changed tasks",
        "",
        f"- Improvements: {', '.join(result.improvements) or 'none'}",
        f"- Regressions: {', '.join(result.regressions) or 'none'}",
        f"- Unstable candidate tasks: {', '.join(result.unstable_candidate_tasks) or 'none'}",
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


def _failures_markdown(result: EvalResult) -> str:
    failed_trials = [trial for trial in result.trials if not trial.passed]
    lines = [
        "# Agent Eval Failure Ledger",
        "",
        "Every row is a failed trial, not a deduplicated task.",
        "",
        "| System | Task | Trial | Failed graders | Output status | Source |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for trial in failed_trials:
        failed_grades = ", ".join(
            grade.name for grade in trial.grades if not grade.passed
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    trial.system_id,
                    trial.task_id,
                    str(trial.trial_index),
                    failed_grades,
                    trial.output.status,
                    trial.output.source or "-",
                ]
            )
            + " |"
        )
    if not failed_trials:
        lines.append("| - | - | - | none | - | - |")
    lines.append("")
    return "\n".join(lines)
