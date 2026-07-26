from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_lab import ContractError, load_contract, run_baseline
from agent_lab.reporting import write_reports


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "contracts" / "agent-system-card.json"
TASKS_PATH = ROOT / "datasets" / "tasks.jsonl"
KNOWLEDGE_PATH = ROOT / "fixtures" / "knowledge" / "product-handbook.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agent Reliability Lab.")
    parser.add_argument(
        "command",
        choices=["check-contract", "baseline"],
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
        "--output",
        type=Path,
        default=ROOT / "reports",
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

    result = run_baseline(args.tasks, args.knowledge, threshold=args.threshold)
    json_path, markdown_path = write_reports(result, args.output)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nReports: {json_path} | {markdown_path}")
    if result.passed != result.total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
