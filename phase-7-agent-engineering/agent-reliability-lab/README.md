# Agent Reliability Lab

This lab turns the Phase 6 enterprise knowledge-base Agent into a brownfield
reliability project. The goal is not to produce the most autonomous demo. The
goal is to make every change measurable against a stable task contract.

Version `0.1.0` contains no model calls and requires no API key. It starts with
the control group that later Agent versions must beat.

## Why start with a non-Agent baseline

Before adding a model, tools, memory, or a graph, we need to answer two
questions:

1. What real job is the system responsible for?
2. Does that job require autonomous decisions at all?

The baseline uses deterministic paragraph retrieval over a small knowledge
fixture. It is intentionally limited, but reproducible. A later Agent version
only counts as progress when it improves the task results without violating
the same boundaries or hiding additional cost.

## Run the checkpoint

Requirements:

- Python 3.10+
- no third-party packages
- no API key

```bash
python run_lab.py check-contract
python run_lab.py baseline
python -m unittest discover -s tests -v
```

Two deliberate failure experiments make the boundary visible:

```bash
python run_lab.py check-contract --contract examples/invalid-card.json
python run_lab.py baseline --threshold 0.0 --output reports/threshold-zero
python run_lab.py baseline --threshold 1.0 --output reports/threshold-one
```

The invalid card and both threshold runs are expected to exit non-zero. A zero
threshold over-answers the unknown expense-policy question; a threshold of one
over-abstains on answerable questions.

The baseline command writes:

```text
reports/
  local/
    baseline.json
    baseline.md
```

The command exits with status `1` when any task fails. That makes the lab
suitable for local verification and CI. Local reports are ignored by Git so
machine-specific latency values do not dirty the checkpoint.

## Read the evidence

Start with these files:

| File | Question it answers |
| --- | --- |
| `contracts/agent-system-card.json` | What job, boundary, failure state, and evidence define this system? |
| `datasets/tasks.jsonl` | Which concrete tasks must remain reproducible? |
| `fixtures/knowledge/product-handbook.md` | What information is the system allowed to use? |
| `agent_lab/baseline.py` | What can a deterministic control group already accomplish? |
| `reports/baseline.json` | Which cases passed, failed, or abstained? |

## Planned checkpoints

Tags are created only when the matching article and code are released.

| Planned tag | Article | Main code delta | Verification |
| --- | --- | --- | --- |
| `ae-01-contract` | Agent System Contract | System Card, task unit, non-Agent baseline | `python run_lab.py baseline` |
| `ae-02-evals` | Agent Evals | graders, repeated runs, comparison report | `python run_lab.py eval` |
| `ae-03-context` | Context Architecture | Context Packet and budget policy | `python run_lab.py context-eval` |
| `ae-04-harness` | Harness Engineering | replaceable runtime boundaries | `python run_lab.py harness-eval` |
| `ae-05-tools` | Tool Engineering | tool registry, dry-run, structured errors | `python run_lab.py tool-eval` |
| `ae-06-loop` | Durable Loop | checkpoints, retries, resume, cancellation | `python run_lab.py fault-test` |
| `ae-07-trace` | Agent Tracing | vendor-neutral spans and redaction | `python run_lab.py trace-review` |
| `ae-08-memory` | Memory Engineering | provenance, TTL, conflict and delete rules | `python run_lab.py memory-eval` |
| `ae-09-graph` | Graph Engineering | dependency graph and independent verifier | `python run_lab.py graph-eval` |
| `ae-10-security` | Human Control | approval policy, sandbox and audit | `python run_lab.py policy-test` |
| `ae-11-ops` | Production Ops | SLO, budgets, degradation and canary | `python run_lab.py ops-game-day` |
| `ae-12-improvement` | Continuous Improvement | failure-to-eval-to-release loop | `python run_lab.py release-gate` |

Commands after `ae-01-contract` are interface targets for future checkpoints,
not features already implemented in `0.1.0`.

## Design rules

- Keep the contract and task IDs stable unless a change is explicitly reviewed.
- Add one main variable per checkpoint.
- Store enough evidence to explain failures, not just the final answer.
- Keep a deterministic path so readers can finish the tutorial without paid
  model calls.
- Treat a real provider as an adapter, not as the definition of the system.
- Do not claim a topology, model, or prompt is better without an equal-budget
  comparison.
