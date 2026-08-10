# Agent CLI Engineering Lab

This phase accompanies the two-part Agent CLI Engineering tutorial. It keeps the domain logic deterministic so the process boundary, output contract, and CLI/MCP adapters can be tested without an API key.

## Requirements

- Python 3.10+
- Codex CLI for the optional real probe
- `mcp==2.0.0` only for the MCP adapter

## Quick start

```bash
cd phase-8-agent-cli-engineering
python -m unittest discover -s tests -v
python run_lab.py runtime-demo --output reports/runtime-demo.json
python -m agent_cli_lab.cli runs list --limit 2 --format json
python -m agent_cli_lab.cli runs get run-003 --format json
python -m agent_cli_lab.cli reports export run-003 --output reports/run-003.json
```

Run the real Codex probe only after `codex --version` and authentication work:

```bash
python run_lab.py codex-probe --mode read-only --output reports/codex-probe.json
```

The probe creates a disposable Git repository, runs `codex exec` with `read-only`, `--ephemeral`, `--ignore-user-config`, JSONL events, and a final JSON Schema. It never uses the dangerous sandbox bypass.

## MCP adapter

Install the optional dependency in an isolated environment:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install "mcp==2.0.0"
.venv/Scripts/python -m agent_cli_lab.mcp_server
```

The MCP server exports `list_runs`, `get_run`, and `export_report`. These call the same functions as the CLI adapter.

## Checkpoints

- `agent-cli-01-runtime`: process runner and real Codex probe
- `agent-cli-02-capabilities`: agent-friendly CLI and MCP 2026-07-28 adapter

