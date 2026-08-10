from __future__ import annotations

from pathlib import Path
from typing import Any

from .records import export_report as export_run_report
from .records import get_run as get_run_record
from .records import list_runs as list_run_records

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as error:  # pragma: no cover - exercised without the optional extra
    raise SystemExit('Install the optional dependency first: pip install "mcp==2.0.0"') from error


server = MCPServer(
    "agent-cli-lab",
    description="Read deterministic Agent run records through MCP 2026-07-28.",
    version="0.2.0",
)


@server.tool(structured_output=True)
def list_runs(limit: int = 10) -> dict[str, Any]:
    """List recent run records. Limit must be between 1 and 100."""
    return list_run_records(limit)


@server.tool(structured_output=True)
def get_run(run_id: str) -> dict[str, Any]:
    """Return one run record selected by its stable ID."""
    return get_run_record(run_id)


@server.tool(structured_output=True)
def export_report(run_id: str, output: str) -> dict[str, Any]:
    """Export one run to an explicit local JSON path and return artifact metadata."""
    return export_run_report(run_id, Path(output))


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()

