"""
Run one question through the real LangGraph Agentic RAG workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_rag_graph import build_agentic_rag_app, build_resources, initial_state, total_latency_ms


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Agentic RAG question")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--json", action="store_true", help="Print raw JSON state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resources = build_resources()
    app = build_agentic_rag_app(resources)
    result = app.invoke(initial_state(args.question))

    if args.json:
        console.print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    console.print(Panel(result["answer"], title="Agentic RAG Answer", border_style="green"))
    table = Table(title="Graph Trace")
    table.add_column("Step")
    table.add_column("Value")
    for index, step in enumerate(result.get("route_trace", []), start=1):
        table.add_row(str(index), step)
    console.print(table)
    console.print(f"[dim]Faithfulness: {result.get('faithfulness', 0):.3f}[/dim]")
    console.print(f"[dim]Latency: {total_latency_ms(result):.0f}ms[/dim]")
    console.print(f"[dim]Cost: ${result.get('llm_usage', {}).get('cost_usd', 0):.4f}[/dim]")


if __name__ == "__main__":
    main()

