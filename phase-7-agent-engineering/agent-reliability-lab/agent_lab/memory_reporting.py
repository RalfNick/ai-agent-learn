from __future__ import annotations

import json
from pathlib import Path

from .memory import MemoryReviewResult


def write_memory_reports(
    result: MemoryReviewResult,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "memory-review.json"
    markdown_path = output_dir / "memory-review.md"
    decisions_path = output_dir / "memory-decisions.jsonl"
    store_path = output_dir / "memory-store.jsonl"
    recall_path = output_dir / "memory-recall.md"

    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_review_markdown(result), encoding="utf-8")
    decisions_path.write_text(_decisions_jsonl(result), encoding="utf-8")
    store_path.write_text(_store_jsonl(result), encoding="utf-8")
    recall_path.write_text(_recall_markdown(result), encoding="utf-8")
    return json_path, markdown_path, decisions_path, store_path, recall_path


def _review_markdown(result: MemoryReviewResult) -> str:
    gate = "PASS" if result.gate_passed else "FAIL"
    lines = [
        "# Agent Memory Review",
        "",
        f"- Version: `{result.version}`",
        f"- Review gate: **{gate}**",
        f"- Matched decisions: `{result.matched_cases}/{result.total_cases}`",
        "- Scope: deterministic policy fixtures, not a model-quality benchmark",
        "",
        "## Decision summary",
        "",
        "| Action | Count |",
        "| --- | ---: |",
    ]
    for action, count in result.decision_counts.items():
        lines.append(f"| `{action}` | {count} |")
    lines.extend(
        [
            "",
            "## Review gate",
            "",
            "| Check | Result |",
            "| --- | --- |",
        ]
    )
    for check, passed in result.gate_checks.items():
        lines.append(f"| `{check}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Case decisions",
            "",
            "| Case | Operation | Expected | Actual | Result |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in result.cases:
        lines.append(
            f"| `{case.case_id}` | `{case.operation}` | "
            f"`{case.expected_action}` | `{case.decision.action}` | "
            f"{'PASS' if case.matched else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "A passing report means this fixture set obeyed the declared write, "
            "recall, conflict, isolation, and deletion policy. It does not prove "
            "that persistent memory improves a live model's answers.",
            "",
        ]
    )
    return "\n".join(lines)


def _decisions_jsonl(result: MemoryReviewResult) -> str:
    lines = [
        json.dumps(
            case.to_dict(purge_ids=result.purge_ids),
            ensure_ascii=False,
            sort_keys=True,
        )
        for case in result.cases
    ]
    return "\n".join(lines) + "\n"


def _store_jsonl(result: MemoryReviewResult) -> str:
    lines = [
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        for record in result.final_store
    ]
    return ("\n".join(lines) + "\n") if lines else ""


def _recall_markdown(result: MemoryReviewResult) -> str:
    recalls = [case for case in result.cases if case.operation == "recall"]
    lines = [
        "# Memory Recall Review",
        "",
        "Recall applies namespace isolation before relevance. Only active, "
        "unexpired records may be returned.",
        "",
        "| Case | Decision | Recalled IDs |",
        "| --- | --- | --- |",
    ]
    for case in recalls:
        recalled = ", ".join(case.decision.recalled_ids) or "none"
        lines.append(
            f"| `{case.case_id}` | `{case.decision.action}` | `{recalled}` |"
        )
    lines.extend(
        [
            "",
            "The cross-tenant fixture returning `none` is the intended result, "
            "even when another tenant has a highly relevant preference.",
            "",
        ]
    )
    return "\n".join(lines)
