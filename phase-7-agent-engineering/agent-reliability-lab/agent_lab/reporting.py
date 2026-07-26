from __future__ import annotations

import json
from pathlib import Path

from .baseline import BaselineResult


def write_reports(result: BaselineResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline.json"
    markdown_path = output_dir / "baseline.md"
    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_to_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def _to_markdown(result: BaselineResult) -> str:
    rows = [
        "# Agent Reliability Lab Baseline",
        "",
        f"- Version: `{result.version}`",
        f"- Strategy: `{result.strategy}`",
        f"- Task pass rate: `{result.task_pass_rate:.0%}`",
        f"- Correct abstention rate: `{result.correct_abstention_rate:.0%}`",
        "",
        "| Task | Status | Score | Passed |",
        "| --- | --- | ---: | --- |",
    ]
    rows.extend(
        f"| `{case.task_id}` | `{case.status}` | {case.score:.2f} | "
        f"{'yes' if case.passed else 'no'} |"
        for case in result.cases
    )
    rows.extend(
        [
            "",
            "This report describes a deterministic control group. It does not "
            "claim that an Agent implementation is already better.",
            "",
        ]
    )
    return "\n".join(rows)
