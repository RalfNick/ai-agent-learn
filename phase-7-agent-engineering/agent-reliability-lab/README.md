# Agent Reliability Lab

This lab turns the Phase 6 enterprise knowledge-base Agent into a brownfield
reliability project. The goal is not to produce the most autonomous demo. The
goal is to make every change measurable against a stable task contract.

Version `1.1.1` contains no live model calls and requires no API key. It preserves
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

The Agent Tracing checkpoint adds a small vendor-neutral span schema and a
review gate around the same ticket-follow-up workflow. Eight fixtures make the
evidence boundary visible: a clean run, wrong context, a safe retry, worker
resume, missing versions, an orphan span, an unclosed span, and a simulated
secret leak. The accepted JSONL output stores source IDs, hashes, lengths,
status, usage, and version evidence instead of raw prompts, email addresses,
ticket bodies, or tokens.

The Memory Engineering checkpoint adds a policy boundary around information
that may outlive one run. Eight fixtures separate current context, resumable
run state, session history, long-term memory, and the business source of truth.
The store accepts explicit preferences and reviewer-verified lessons, routes
ticket facts back to their owning system, rejects inference and sensitive
values, isolates recall by namespace, supersedes corrections, and purges
statements on deletion while retaining content-free audit evidence.

The Graph Engineering checkpoint adds a small DAG compiler and execution review
around six boundary fixtures. It accepts a valid Diamond workflow, rejects
missing dependencies, cycles, and parallel state-write conflicts, blocks merge
when an independent verifier fails, and stops before overspending its declared
budget. The experiment evaluates control behavior only; it does not claim that
a graph improves live-model quality, latency, or cost.

The Human Control checkpoint separates model proposals from deterministic tool
policy, approval state, credential-bearing execution, and audit evidence. Eight
fixtures cover read-only and bounded reversible actions, approved and rejected
external sends, approval expiry, changed arguments, an unauthorized production
reviewer, and duplicate resume. Approval envelopes bind the exact normalized
action, and exported reports retain hashes and receipts instead of message bodies
or credentials.

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
python run_lab.py trace-review
python run_lab.py memory-review
python run_lab.py graph-eval
python run_lab.py policy-test
python run_lab.py ops-loop
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
python run_lab.py trace-review --output reports/trace-local
python run_lab.py memory-review --output reports/memory-local
python run_lab.py graph-eval --output reports/graph-local
python run_lab.py policy-test --output reports/security-local
python run_lab.py ops-loop --output reports/operations-local
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

The trace review is also expected to pass. Passing means that the review gate
matches every fixture's declared finding, removes the simulated sensitive value
before export, and can answer more of this lab's five debugging questions than
the deliberately sparse one-line-log control. It is not a claim that structured
tracing improves every system by a universal percentage.

The memory review is expected to pass when all eight declared decisions match,
tenant isolation happens before relevance, only one version per key remains
active, and deleted statements are absent from every exported report. It does
not test semantic retrieval quality or claim that memory improves a live model.

The graph evaluation is expected to pass when all six declared terminal states
match, invalid graphs are rejected before execution, a failed verifier prevents
merge, parallel branches own separate state keys, and the budget fixture stops
without overspending. A passing gate is not evidence that more agents are better.

The policy test is expected to pass when all eight declared outcomes match,
mutating external actions produce no effect before approval, rejected, expired,
changed, and unauthorized paths stay effect-free, and duplicate resume replays a
receipt without performing the action twice. It is not a penetration test or an
audit of a real IAM deployment.

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
    trace-review.json
    trace-review.md
    trace-failures.md
    traces.jsonl
    memory-review.json
    memory-review.md
    memory-decisions.jsonl
    memory-store.jsonl
    memory-recall.md
    graph-review.json
    graph-review.md
    graph-runs.jsonl
    graph-failures.md
    security-review.json
    security-review.md
    security-runs.jsonl
    security-failures.md
    operations-review.json
    operations-review.md
    operations-runs.jsonl
    incident-evals.jsonl
    operations-failures.md
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
| `datasets/trace-cases.jsonl` | Which tree, context, retry, version, terminal-state, and privacy cases define trace review behavior? |
| `datasets/memory-cases.jsonl` | Which write, source-of-truth, inference, isolation, conflict, expiry, and deletion cases define Memory behavior? |
| `datasets/graph-cases.jsonl` | Which dependency, cycle, state ownership, verifier, merge, and budget cases define Graph behavior? |
| `datasets/security-cases.jsonl` | Which risk class, approval, expiry, reviewer, action-binding, and idempotency cases define Human Control? |
| `datasets/operations-cases.jsonl` | Which SLO, budget, degradation, recovery, canary, rollback, and incident-to-eval cases define the production loop? |
| `fixtures/knowledge/product-handbook.md` | What information is the system allowed to use? |
| `fixtures/context/context-sources.jsonl` | Which trusted, expired, restricted, noisy, or untrusted sources can enter a packet? |
| `agent_lab/baseline.py` | What can a deterministic control group already accomplish? |
| `agent_lab/evals.py` | How are tasks, trials, graders, summaries, and release gates implemented? |
| `agent_lab/context.py` | How are Context Packets selected, filtered, budgeted, rendered, and graded? |
| `agent_lab/harness.py` | Which component owns the model seam, tool dispatch, policy, session, stop, resume, verification, and trace? |
| `agent_lab/tools.py` | How are model-facing schemas separated from runtime permission, approval, idempotency, retry, and output contracts? |
| `agent_lab/durable.py` | How are checkpoints, stable actions, retry policy, reconciliation, cancellation, leases, and fencing implemented? |
| `agent_lab/tracing.py` | How are spans built, sanitized, linked, checked, and used to answer five debugging questions? |
| `agent_lab/memory.py` | How are candidates evaluated, namespaced, recalled, superseded, expired, and deleted? |
| `agent_lab/graph.py` | How are dependencies compiled, parallel writes isolated, budgets enforced, and verified branches merged? |
| `agent_lab/security.py` | How are tool risks classified, approvals bound, decisions resumed, credentials isolated, and effects made idempotent? |
| `agent_lab/operations.py` | How do production signals choose continue, degrade, pause, rollback, promote, and regression-eval actions? |
| `reports/baseline.json` | Which cases passed, failed, or abstained? |
| `reports/trials.jsonl` | What happened in every repeated attempt? |
| `reports/context-packets.jsonl` | What each strategy selected, excluded, marked missing, and spent? |
| `reports/harness-runs.jsonl` | Which state and event sequence each Harness case produced? |
| `reports/tool-runs.jsonl` | Which fixed call proposals passed each tool-contract grader and which side effects occurred? |
| `reports/durable-runs.jsonl` | Which persisted state and event sequence each injected fault produced? |
| `reports/traces.jsonl` | Which sanitized spans reconstruct each fixture's causal execution path? |
| `reports/trace-review.md` | Which debugging questions are answerable and whether the trace review gate passed? |
| `reports/memory-review.md` | Which Memory decisions matched and whether the lifecycle review gate passed? |
| `reports/memory-recall.md` | Did namespace-first recall prevent the cross-tenant fixture from leaking a preference? |
| `reports/graph-review.md` | Did all six Graph control outcomes and release checks match? |
| `reports/security-review.md` | Did all eight Human Control outcomes and release checks match? |
| `reports/operations-review.md` | Did all nine production windows and release checks match? |

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
| `ae-08-memory` | Memory Engineering | provenance, TTL, conflict and delete rules | `python run_lab.py memory-review` |
| `ae-09-graph` | Graph Engineering | dependency graph and independent verifier | `python run_lab.py graph-eval` |
| `ae-10-security` | Human Control | approval policy, sandbox and audit | `python run_lab.py policy-test` |
| `ae-11-production-loop` | Production Loop | SLO, budgets, degradation, canary and incident-to-eval loop | `python run_lab.py ops-loop` |

The `ae-11-production-loop` checkpoint merges the originally planned Production
Ops and Continuous Improvement articles into one executable loop.

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

## Agent Tracing experiment limits

The trace review uses deterministic logical sequence numbers and durations. It
does not measure clock skew, exporter latency, sampling, Collector throughput,
or storage cost. The five-question comparison is a teaching fixture, not a
general benchmark. The lab records observable inputs, source IDs, calls,
statuses, errors, and versions; it does not expose hidden model reasoning. Use
OpenTelemetry or a dedicated observability backend when traces must cross
processes, services, environments, or retention boundaries.

## Memory Engineering experiment limits

The memory review is a deterministic policy exercise over an in-memory store.
Token overlap is used only to make recall order inspectable; it is not a stand-in
for a production semantic index. The lab does not test database encryption,
retention jobs, concurrent writers, regional residency, provider-side session
storage, or legal erasure guarantees. A passing gate proves the eight fixtures
obey the declared policy, not that the model learned, changed its weights, or
will answer better because persistent state exists.

## Production Loop experiment limits

The operations review evaluates nine pre-sanitized metadata windows. It emits
policy decisions such as `read_only`, `rollback`, and `promote`; it does not
shift traffic, deploy versions, redact raw traces, deduplicate an eval store, or
compensate completed side effects. A passing gate proves the declared fixtures
match the policy, not that the thresholds or rollback path are production-ready.
