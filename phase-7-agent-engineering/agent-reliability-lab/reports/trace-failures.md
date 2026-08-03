# Trace Finding Ledger

These fixtures are intentionally mixed: a finding is correct when it matches the case contract.

| Case | Finding | Span | Detail | Expected match |
| --- | --- | --- | --- | --- |
| clean-run | none | - | clean trace | yes |
| wrong-context | context_mismatch | retrieve | selected sources do not satisfy the task contract | yes |
| safe-retry | none | - | clean trace | yes |
| worker-resume | none | - | clean trace | yes |
| missing-version | missing_version_evidence | model | missing prompt | yes |
| orphan-span | orphan_span | tool | parent missing-parent does not exist | yes |
| unclosed-span | missing_terminal_status | model | status 'unset' is not terminal | yes |
| secret-leak | sensitive_attribute | tool | sensitive_value:result_code | yes |
