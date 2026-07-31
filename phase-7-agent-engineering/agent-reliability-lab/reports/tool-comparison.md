# Tool Contract Comparison

- Tool eval version: `0.5.0`
- Release gate: **PASS**
- Scope: fixed proposed calls exercise runtime contracts; this is not a model tool-selection benchmark.

| Metric | wide-tool-v1 | typed-registry-v2 |
| --- | ---: | ---: |
| Case pass rate | 9.1% | 100.0% |
| Unsafe side effects | 4 | 0 |
| Duplicate side effects | 2 | 0 |
| Actionable error rate | 0.0% | 100.0% |
| Model-facing schema bytes | 247 | 2505 |

The typed catalog is intentionally larger. Its safety and recovery metadata are useful, but the additional context cost is not free.

## Changed cases

- Improvements: preview-before-write, write-needs-approval, invalid-arguments, permission-boundary, duplicate-action, idempotency-conflict, transient-timeout, bounded-list, ticket-not-found, invalid-cursor
- Regressions: none

## Release gate

- [x] `candidate_passes_all_cases`
- [x] `candidate_blocks_unsafe_side_effects`
- [x] `candidate_deduplicates_writes`
- [x] `candidate_errors_are_actionable`
- [x] `no_case_regressions`
