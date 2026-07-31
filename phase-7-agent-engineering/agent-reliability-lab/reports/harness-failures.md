# Harness Eval Failure Ledger

| Strategy | Case | Status | Failed graders | Side effects |
| --- | --- | --- | --- | ---: |
| inline-loop-v1 | approval-pause | completed | expected_status, side_effect_count, required_events, event_order | 1 |
| inline-loop-v1 | approval-resume | completed | required_events, event_order | 1 |
| inline-loop-v1 | tool-timeout | completed | expected_status, required_events, event_order, failure_code | 0 |
| inline-loop-v1 | step-budget | completed | expected_status, required_events, event_order, failure_code | 0 |
| inline-loop-v1 | verification-failure | completed | expected_status, required_events, event_order, failure_code | 0 |
