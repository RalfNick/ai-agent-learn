# Human Control Stop Review

These stopped paths are expected policy outcomes, not hidden successes.

| Case | Status | Reason | Effects |
| --- | --- | --- | ---: |
| `external-rejected` | `rejected` | `rejected_by_reviewer` | 0 |
| `approval-expired` | `expired` | `approval_expired` | 0 |
| `arguments-changed-after-approval` | `blocked` | `action_changed` | 0 |
| `critical-wrong-reviewer` | `denied` | `reviewer_not_authorized` | 0 |
