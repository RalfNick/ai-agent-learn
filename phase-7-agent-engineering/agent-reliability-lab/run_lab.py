from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_lab import load_contract, run_baseline
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
    contract = load_contract(CONTRACT_PATH)
    if args.command == "check-contract":
        print(
            json.dumps(
                {
                    "status": "valid",
                    "id": contract["id"],
                    "version": contract["version"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    result = run_baseline(TASKS_PATH, KNOWLEDGE_PATH)
    json_path, markdown_path = write_reports(result, args.output)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nReports: {json_path} | {markdown_path}")
    if result.passed != result.total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
