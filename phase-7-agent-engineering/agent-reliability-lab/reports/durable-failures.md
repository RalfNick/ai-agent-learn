# Durable Loop Failure Ledger

| Strategy | Case | Failed graders | Effects | Duplicates | Final state |
| --- | --- | --- | ---: | ---: | --- |
| process-loop-v1 | restart-after-model | model_attempts, required_events | 1 | 0 | completed |
| process-loop-v1 | model-transient-retry | required_events | 1 | 0 | completed |
| process-loop-v1 | model-permanent-error | model_attempts, failure_code, required_events | 0 | 0 | failed |
| process-loop-v1 | retry-budget-exhausted | required_events | 0 | 0 | failed |
| process-loop-v1 | write-receipt-recovery | side_effect_count, duplicate_side_effects, required_events | 2 | 1 | completed |
| process-loop-v1 | write-unknown | expected_status, side_effect_count, failure_code, required_events | 1 | 0 | completed |
| process-loop-v1 | cancel-at-human-wait | expected_status, side_effect_count, failure_code, required_events | 1 | 0 | completed |
| process-loop-v1 | stale-worker | side_effect_count, duplicate_side_effects, required_events | 2 | 1 | completed |
