# Tool Eval Failure Ledger

| Strategy | Case | Failed graders | Side effects | Final error |
| --- | --- | --- | ---: | --- |
| wide-tool-v1 | preview-before-write | side_effect_count, output_keys | 1 | - |
| wide-tool-v1 | write-needs-approval | ok, side_effect_count, error_code, retryable | 1 | - |
| wide-tool-v1 | invalid-arguments | ok, side_effect_count, error_code, retryable | 1 | - |
| wide-tool-v1 | permission-boundary | ok, side_effect_count, error_code, retryable | 1 | - |
| wide-tool-v1 | duplicate-action | side_effect_count, replayed, output_keys | 2 | - |
| wide-tool-v1 | idempotency-conflict | ok, side_effect_count, error_code, retryable | 2 | - |
| wide-tool-v1 | transient-timeout | error_code, retryable | 0 | tool_failed |
| wide-tool-v1 | bounded-list | output_keys, output_count, has_next_cursor | 0 | - |
| wide-tool-v1 | ticket-not-found | ok, error_code, retryable | 0 | - |
| wide-tool-v1 | invalid-cursor | ok, error_code, retryable | 0 | - |
