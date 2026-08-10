from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .records import RunNotFoundError, export_report, get_run, list_runs


def _emit(data: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    if "items" in data:
        for item in data["items"]:
            print(f"{item['id']}\t{item['status']}\t{item['task']}")
        return
    for key, value in data.items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-lab", description="Query deterministic Agent run records.")
    groups = parser.add_subparsers(dest="group", required=True)

    runs = groups.add_parser("runs", help="List or inspect runs.")
    run_commands = runs.add_subparsers(dest="command", required=True)
    list_command = run_commands.add_parser("list", help="List recent runs.")
    list_command.add_argument("--limit", type=int, default=10)
    list_command.add_argument("--format", choices=("text", "json"), default="text")
    get_command = run_commands.add_parser("get", help="Get one run by stable ID.")
    get_command.add_argument("run_id")
    get_command.add_argument("--format", choices=("text", "json"), default="text")

    reports = groups.add_parser("reports", help="Export a run as a file artifact.")
    report_commands = reports.add_subparsers(dest="command", required=True)
    export_command = report_commands.add_parser("export", help="Export one run to JSON.")
    export_command.add_argument("run_id")
    export_command.add_argument("--output", type=Path, required=True)
    export_command.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.group == "runs" and args.command == "list":
            data = list_runs(args.limit)
            output_format = args.format
        elif args.group == "runs" and args.command == "get":
            data = get_run(args.run_id)
            output_format = args.format
        else:
            data = export_report(args.run_id, args.output)
            output_format = args.format
        _emit(data, output_format)
        return 0
    except (ValueError, RunNotFoundError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

