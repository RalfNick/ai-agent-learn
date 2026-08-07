# Agent Graph Review

- Version: `graph-control-v1`
- Review gate: **PASS**
- Matched outcomes: `6/6`
- Scope: deterministic graph-control fixtures, not a model or topology benchmark

## Status summary

| Status | Count |
| --- | ---: |
| `blocked` | 2 |
| `completed` | 1 |
| `invalid` | 3 |

## Release gate

| Check | Result |
| --- | --- |
| `expected_results` | PASS |
| `invalid_graphs_rejected` | PASS |
| `merge_requires_verification` | PASS |
| `parallel_state_isolated` | PASS |
| `independent_verifier` | PASS |
| `budget_honored` | PASS |
| `trace_complete` | PASS |

## Case outcomes

| Case | Expected | Actual | Reason | Budget | Merge |
| --- | --- | --- | --- | ---: | --- |
| `valid-diamond` | `completed` | `completed` | `completed` | 6/6 | yes |
| `missing-dependency` | `invalid` | `invalid` | `missing_dependency` | 0/2 | no |
| `cycle-detected` | `invalid` | `invalid` | `cycle_detected` | 0/2 | no |
| `shared-write-conflict` | `invalid` | `invalid` | `shared_write_conflict` | 0/3 | no |
| `verifier-blocks-merge` | `blocked` | `blocked` | `verifier_failed` | 5/6 | no |
| `budget-exhausted` | `blocked` | `blocked` | `budget_exhausted` | 3/3 | no |

PASS means the six fixtures matched their declared control outcomes. It does not show that a graph improves live-model quality, latency, or cost.
