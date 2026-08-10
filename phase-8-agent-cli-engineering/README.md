# Agent CLI Engineering Lab

This checkpoint accompanies the first Agent CLI Engineering tutorial. It keeps the process boundary deterministic and provides an optional real Codex probe.

## Requirements

- Python 3.10+
- Codex CLI for the optional real probe

## Quick start

```bash
cd phase-8-agent-cli-engineering
python -m unittest discover -s tests -v
python run_lab.py runtime-demo --output reports/runtime-demo.json
```

Run the real Codex probe only after `codex --version` and authentication work:

```bash
python run_lab.py codex-probe --mode read-only --output reports/codex-probe.json
```

The probe creates a disposable Git repository, runs `codex exec` with `read-only`, `--ephemeral`, `--ignore-user-config`, JSONL events, and a final JSON Schema. It never uses the dangerous sandbox bypass.

## Checkpoints

- `agent-cli-01-runtime`: this checkpoint, containing the process runner and real Codex probe
- `agent-cli-02-capabilities`: the later checkpoint adds the agent-friendly CLI and MCP 2026-07-28 adapter
