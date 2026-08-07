from __future__ import annotations

import json
from pathlib import Path

from .graph import GraphEvalResult


def write_graph_reports(
    result: GraphEvalResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "graph-review.json"
    markdown_path = output_dir / "graph-review.md"
    runs_path = output_dir / "graph-runs.jsonl"
    failures_path = output_dir / "graph-failures.md"

    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_review_markdown(result), encoding="utf-8")
    runs_path.write_text(_runs_jsonl(result), encoding="utf-8")
    failures_path.write_text(_failures_markdown(result), encoding="utf-8")
    return json_path, markdown_path, runs_path, failures_path


def _review_markdown(result: GraphEvalResult) -> str:
    gate = "PASS" if result.gate_passed else "FAIL"
    lines = [
        "# Agent Graph Review",
        "",
        f"- Version: `{result.version}`",
        f"- Review gate: **{gate}**",
        f"- Matched outcomes: `{result.matched_cases}/{result.total_cases}`",
        "- Scope: deterministic graph-control fixtures, not a model or topology benchmark",
        "",
        "## Status summary",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(result.status_counts.items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## Release gate", "", "| Check | Result |", "| --- | --- |"])
    for check, passed in result.gate_checks.items():
        lines.append(f"| `{check}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Case outcomes",
            "",
            "| Case | Expected | Actual | Reason | Budget | Merge |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for case in result.cases:
        lines.append(
            f"| `{case.case_id}` | `{case.expected_status}` | `{case.status}` | "
            f"`{case.reason}` | {case.spent_budget}/{case.budget} | "
            f"{'yes' if case.merge_executed else 'no'} |"
        )
    lines.extend(
        [
            "",
            "PASS means the six fixtures matched their declared control outcomes. "
            "It does not show that a graph improves live-model quality, latency, or cost.",
            "",
        ]
    )
    return "\n".join(lines)


def _runs_jsonl(result: GraphEvalResult) -> str:
    return "\n".join(
        json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True)
        for case in result.cases
    ) + "\n"


def _failures_markdown(result: GraphEvalResult) -> str:
    stopped = [case for case in result.cases if case.status != "completed"]
    lines = [
        "# Graph Stop Review",
        "",
        "These are expected rejected or blocked paths, not hidden successes.",
        "",
        "| Case | Terminal status | Reason | Last event |",
        "| --- | --- | --- | --- |",
    ]
    for case in stopped:
        last_event = case.events[-1].phase if case.events else "none"
        lines.append(
            f"| `{case.case_id}` | `{case.status}` | `{case.reason}` | `{last_event}` |"
        )
    lines.append("")
    return "\n".join(lines)
