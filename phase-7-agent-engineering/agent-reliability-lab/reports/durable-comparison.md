# Durable Loop Fault Comparison

- Durable eval version: `0.6.0`
- Release gate: **PASS**
- Scope: deterministic process-loss and dependency-fault fixtures; not a distributed runtime benchmark.

| Metric | process-loop-v1 | durable-loop-v1 |
| --- | ---: | ---: |
| Case pass rate | 11.1% | 100.0% |
| Total model attempts | 16 | 13 |
| Duplicate side effects | 2 | 0 |
| Blind retries | 8 | 0 |
| Explicit terminal rate | 100.0% | 100.0% |

## Changed cases

- Improvements: cancel-at-human-wait, model-permanent-error, model-transient-retry, restart-after-model, retry-budget-exhausted, stale-worker, write-receipt-recovery, write-unknown
- Regressions: none

## Release gate

- [x] `all_candidate_cases_pass`
- [x] `at_least_one_measured_improvement`
- [x] `no_case_regressions`
- [x] `restart_rehydrates_checkpoint`
- [x] `unknown_committed_write_recovers_receipt`
- [x] `unknown_unconfirmed_write_stops_for_reconciliation`
- [x] `cancel_blocks_later_side_effect`
- [x] `stale_worker_is_fenced`
- [x] `candidate_has_no_duplicate_side_effect`
- [x] `candidate_has_no_blind_retry`
