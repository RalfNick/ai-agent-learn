# Context Eval Failure Ledger

| Strategy | Case | Failed graders | Selected sources | Missing topics |
| --- | --- | --- | --- | --- |
| dump-all-v1 | ctx-001 | forbidden_sources, relevance | general-changelog, external-pii-injection | - |
| dump-all-v1 | ctx-002 | forbidden_sources, relevance | rate-limit-policy-expired, general-changelog | - |
| dump-all-v1 | ctx-005 | relevance | general-changelog | expense-policy |
| dump-all-v1 | ctx-006 | missing_evidence, forbidden_sources | pii-policy-current | - |
| dump-all-v1 | ctx-008 | required_evidence, missing_evidence | - | rollout-status |
