# Graph Stop Review

These are expected rejected or blocked paths, not hidden successes.

| Case | Terminal status | Reason | Last event |
| --- | --- | --- | --- |
| `missing-dependency` | `invalid` | `missing_dependency` | `compile_rejected` |
| `cycle-detected` | `invalid` | `cycle_detected` | `compile_rejected` |
| `shared-write-conflict` | `invalid` | `shared_write_conflict` | `compile_rejected` |
| `verifier-blocks-merge` | `blocked` | `verifier_failed` | `failed` |
| `budget-exhausted` | `blocked` | `budget_exhausted` | `blocked` |
