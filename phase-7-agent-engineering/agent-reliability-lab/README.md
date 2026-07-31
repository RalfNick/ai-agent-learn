# Agent Reliability Lab

This lab turns the Phase 6 enterprise knowledge-base Agent into a brownfield
reliability project. The goal is not to produce the most autonomous demo. The
goal is to make every change measurable against a stable task contract.

Version `0.6.0` contains no live model calls and requires no API key. It preserves
the contract and eval checkpoints, then adds a Context Packet assembler that
compares a naive prefix dump with explicit source, freshness, clearance,
relevance, missing-evidence, and budget policies. The Harness checkpoint then
compares a direct model-to-tool loop with a minimal runtime that owns policy
checks, JSON-serializable run state, approval pause/resume, tool timeouts, step
limits, verification, and event traces.

The Tool Engineering checkpoint adds an explicit registry around the same
ticket backend. It compares one wide, loosely typed operation surface with
separate read, preview, write, paginated-list, and slow-dependency tools. The
registry validates input and output schemas, permission, approval, idempotency,
timeout, retry classification, and bounded output around dispatch.

The Durable Loop checkpoint moves run state from process memory into atomic JSON
checkpoints and tests what happens around uncertain boundaries. Nine fixtures
cover process restart, transient and permanent model failures, retry exhaustion,
a committed write whose response is lost, an unconfirmed write, cancellation
during a human wait, and a stale worker after lease takeover. Stable action IDs,
receipt lookup, explicit reconciliation, persisted cancellation, and fencing
tokens are compared with a deliberately naive process-local loop.

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
python run_lab.py eval
python run_lab.py context-eval
python run_lab.py harness-eval
python run_lab.py tool-eval
python run_lab.py fault-test
python -m unittest discover -s tests -v
```

Two deliberate failure experiments make the boundary visible:

```bash
python run_lab.py check-contract --contract examples/invalid-card.json
python run_lab.py baseline --threshold 0.0 --output reports/threshold-zero
python run_lab.py baseline --threshold 1.0 --output reports/threshold-one
python run_lab.py eval --candidate flaky-simulator --output reports/flaky-demo
python run_lab.py context-eval --context-budget 10 --output reports/context-budget-ten
python run_lab.py harness-eval --max-steps 1 --output reports/harness-step-one
python run_lab.py tool-eval --output reports/tool-local
python run_lab.py fault-test --output reports/durable-local
```

The invalid card, both threshold runs, and the flaky candidate are expected to exit non-zero. A zero
threshold over-answers the unknown expense-policy question; a threshold of one
over-abstains on answerable questions. The flaky simulator changes its behavior
between repeated trials so the stability release gate rejects it. The tiny
context budget cannot retain required evidence, so the context gate rejects it.
The one-step Harness override stops valid multi-step cases too early, so the
Harness release gate rejects that configuration.

The tool comparison is expected to pass because the release gate applies to the
typed registry. The wide-tool control remains in the report so its unsafe
writes, duplicate effect, generic timeout, and unbounded list output stay
visible.

The durable comparison is expected to pass because the release gate applies to
`durable-loop-v1`. The process-local control remains in the report so blind
retries, duplicate writes, lost cancellation, and stale-worker effects stay
visible.

The commands write:

```text
reports/
  local/
    baseline.json
    baseline.md
    comparison.json
    comparison.md
    failures.md
    trials.jsonl
    context-comparison.json
    context-comparison.md
    context-failures.md
    context-packets.jsonl
    harness-comparison.json
    harness-comparison.md
    harness-failures.md
    harness-runs.jsonl
    tool-comparison.json
    tool-comparison.md
    tool-failures.md
    tool-runs.jsonl
    durable-comparison.json
    durable-comparison.md
    durable-failures.md
    durable-runs.jsonl
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
| `datasets/eval-tasks.jsonl` | Which regression, capability, and safety cases compare two versions? |
| `datasets/context-cases.jsonl` | Which source, budget, access, and missing-evidence cases compare two context strategies? |
| `datasets/harness-cases.jsonl` | Which approval, resume, timeout, budget, and verification cases define the runtime boundary? |
| `datasets/tool-cases.jsonl` | Which schema, permission, approval, idempotency, timeout, and pagination cases define each tool boundary? |
| `datasets/durable-cases.jsonl` | Which restart, retry, result-unknown, cancellation, and lease cases define recovery behavior? |
| `fixtures/knowledge/product-handbook.md` | What information is the system allowed to use? |
| `fixtures/context/context-sources.jsonl` | Which trusted, expired, restricted, noisy, or untrusted sources can enter a packet? |
| `agent_lab/baseline.py` | What can a deterministic control group already accomplish? |
| `agent_lab/evals.py` | How are tasks, trials, graders, summaries, and release gates implemented? |
| `agent_lab/context.py` | How are Context Packets selected, filtered, budgeted, rendered, and graded? |
| `agent_lab/harness.py` | Which component owns the model seam, tool dispatch, policy, session, stop, resume, verification, and trace? |
| `agent_lab/tools.py` | How are model-facing schemas separated from runtime permission, approval, idempotency, retry, and output contracts? |
| `agent_lab/durable.py` | How are checkpoints, stable actions, retry policy, reconciliation, cancellation, leases, and fencing implemented? |
| `reports/baseline.json` | Which cases passed, failed, or abstained? |
| `reports/trials.jsonl` | What happened in every repeated attempt? |
| `reports/context-packets.jsonl` | What each strategy selected, excluded, marked missing, and spent? |
| `reports/harness-runs.jsonl` | Which state and event sequence each Harness case produced? |
| `reports/tool-runs.jsonl` | Which fixed call proposals passed each tool-contract grader and which side effects occurred? |
| `reports/durable-runs.jsonl` | Which persisted state and event sequence each injected fault produced? |

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

Commands after `ae-06-loop` are interface targets for future checkpoints, not
features already implemented in `0.6.0`.

## Design rules

- Keep the contract and task IDs stable unless a change is explicitly reviewed.
- Add one main variable per checkpoint.
- Store enough evidence to explain failures, not just the final answer.
- Keep a deterministic path so readers can finish the tutorial without paid
  model calls.
- Treat a real provider as an adapter, not as the definition of the system.
- Do not claim a topology, model, or prompt is better without an equal-budget
  comparison.

## Harness experiment limits

The Harness comparison measures contract conformance in a deterministic
fixture. The `inline-loop-v1` side is a deliberately small control, not a
production SDK. Simulated latency metadata makes timeout behavior reproducible;
it does not test wall-clock cancellation. The in-memory session and idempotency
ledger demonstrate interfaces and event order, not production durability.

## Tool experiment limits

The Tool Engineering comparison replays fixed, inspectable call proposals. It
proves that the candidate runtime enforces its declared contracts; it does not
measure whether a live model selects the right tool. The baseline's ambiguous
preview call is an adversarial fixture, not a measured provider failure. Run a
separate repeated provider evaluation before claiming that names or descriptions
improve model selection accuracy. Schema size is reported in UTF-8 bytes as a
stable local proxy, not as provider-token billing.

## Durable Loop experiment limits

The fault test rebuilds runner objects and restores JSON state from disk, but it
does not kill an operating-system process or run concurrent distributed workers.
Fault timing, backoff time, dependency responses, and lease takeover are
deterministic fixtures. Atomic file replacement demonstrates the persistence
boundary, not production database durability. Use a real durable runtime and
integration tests before making availability or exactly-once claims.
