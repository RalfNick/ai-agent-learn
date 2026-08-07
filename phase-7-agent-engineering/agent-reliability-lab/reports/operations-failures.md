# Production Loop Failure Review

## Contract mismatches

No declared outcome mismatches.

## Controlled incidents

| Window | Action | Reason | Eval task |
| --- | --- | --- | --- |
| `tool-throttle-degrade` | `read_only` | `tool_error_budget_exceeded` | `regression-6dbfd98c73e6` |
| `latency-queue-degrade` | `draft_only` | `latency_slo_missed` | `regression-463ab5d0aa6d` |
| `task-budget-stop` | `handoff` | `task_cost_budget_exceeded` | `regression-74928b75b3a3` |
| `unknown-write-pause` | `pause_writes` | `write_outcome_unknown` | `regression-db26e1e492b2` |
| `provider-unavailable` | `handoff` | `model_provider_unavailable` | `regression-d7db05452e7d` |
| `canary-regression` | `rollback` | `canary_error_regression` | `regression-f1437ebd06c4` |
