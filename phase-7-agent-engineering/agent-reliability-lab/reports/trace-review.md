# Agent Trace Review

- Trace checkpoint: `0.7.0`
- Review gate: **PASS**
- Scope: deterministic trace-contract fixtures; not a production observability benchmark.

## Comparison

| Evidence source | Debugging question answer rate |
| --- | ---: |
| one-line log | 20.0% |
| structured trace | 97.5% |

## Five debugging questions

| Case | Context sources | Tool path | First failure | Retry evidence | Versions |
| --- | --- | --- | --- | --- | --- |
| clean-run | yes | yes | yes | yes | yes |
| wrong-context | yes | yes | yes | yes | yes |
| safe-retry | yes | yes | yes | yes | yes |
| worker-resume | yes | yes | yes | yes | yes |
| missing-version | yes | yes | yes | yes | no |
| orphan-span | yes | yes | yes | yes | yes |
| unclosed-span | yes | yes | yes | yes | yes |
| secret-leak | yes | yes | yes | yes | yes |

## Review gate

- [x] `expected_findings_matched`
- [x] `clean_trace_has_no_findings`
- [x] `sensitive_values_removed`
- [x] `safe_retry_has_evidence`
- [x] `more_answerable_than_plain_log`
