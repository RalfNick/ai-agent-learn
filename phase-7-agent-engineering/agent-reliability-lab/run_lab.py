from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_lab import (
    ContractError,
    load_contract,
    run_baseline,
    run_context_eval,
    run_eval,
)
from agent_lab.context_reporting import write_context_reports
from agent_lab.eval_reporting import write_eval_reports
from agent_lab.reporting import write_reports


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "contracts" / "agent-system-card.json"
TASKS_PATH = ROOT / "datasets" / "tasks.jsonl"
EVAL_TASKS_PATH = ROOT / "datasets" / "eval-tasks.jsonl"
CONTEXT_CASES_PATH = ROOT / "datasets" / "context-cases.jsonl"
KNOWLEDGE_PATH = ROOT / "fixtures" / "knowledge" / "product-handbook.md"
CONTEXT_SOURCES_PATH = (
    ROOT / "fixtures" / "context" / "context-sources.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agent Reliability Lab.")
    parser.add_argument(
        "command",
        choices=["check-contract", "baseline", "eval", "context-eval"],
        help="Validation or baseline command to run.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=CONTRACT_PATH,
        help="Agent System Card to validate before running.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=TASKS_PATH,
        help="JSONL task dataset used by the baseline.",
    )
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=KNOWLEDGE_PATH,
        help="Markdown knowledge fixture used by the baseline.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.28,
        help="Minimum retrieval score required to answer (default: 0.28).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of attempts per task for eval runs (default: 3).",
    )
    parser.add_argument(
        "--candidate",
        choices=["candidate-v2", "flaky-simulator"],
        default="candidate-v2",
        help="Candidate system used by eval (default: candidate-v2).",
    )
    parser.add_argument(
        "--context-cases",
        type=Path,
        default=CONTEXT_CASES_PATH,
        help="JSONL context assembly cases.",
    )
    parser.add_argument(
        "--context-sources",
        type=Path,
        default=CONTEXT_SOURCES_PATH,
        help="JSONL context source catalog.",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=None,
        help="Optional estimated-token budget override for every context case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "local",
        help="Directory for baseline reports.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        contract = load_contract(args.contract)
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "invalid",
                    "path": str(args.contract),
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from exc
    if args.command == "check-contract":
        print(
            json.dumps(
                {
                    "status": "valid",
                    "path": str(args.contract),
                    "id": contract["id"],
                    "version": contract["version"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0.0 and 1.0")

    if args.command == "eval":
        tasks_path = args.tasks if args.tasks != TASKS_PATH else EVAL_TASKS_PATH
        result = run_eval(
            tasks_path,
            args.knowledge,
            threshold=args.threshold,
            trials_per_task=args.trials,
            candidate_id=args.candidate,
        )
        json_path, markdown_path, failures_path, trials_path = write_eval_reports(
            result, args.output
        )
        print(json.dumps(result.summary_dict(), ensure_ascii=False, indent=2))
        print(
            "\nReports: "
            f"{json_path} | {markdown_path} | {failures_path} | {trials_path}"
        )
        if not result.gate_passed:
            raise SystemExit(1)
        return

    if args.command == "context-eval":
        result = run_context_eval(
            args.context_cases,
            args.context_sources,
            budget_override=args.context_budget,
        )
        json_path, markdown_path, failures_path, packets_path = (
            write_context_reports(result, args.output)
        )
        print(json.dumps(result.summary_dict(), ensure_ascii=False, indent=2))
        print(
            "\nReports: "
            f"{json_path} | {markdown_path} | {failures_path} | {packets_path}"
        )
        if not result.gate_passed:
            raise SystemExit(1)
        return

    result = run_baseline(args.tasks, args.knowledge, threshold=args.threshold)
    json_path, markdown_path = write_reports(result, args.output)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nReports: {json_path} | {markdown_path}")
    if result.passed != result.total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
