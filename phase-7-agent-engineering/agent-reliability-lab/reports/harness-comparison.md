# Harness Boundary Comparison

- Harness eval version: `0.4.0`
- Release gate: **PASS**
- Scope: deterministic boundary conformance; not model quality or SDK ranking.

| Metric | inline-loop-v1 | minimal-harness-v1 |
| --- | ---: | ---: |
| Case pass rate | 16.7% | 100.0% |
| Explicit state rate | 100.0% | 100.0% |
| Sensitive-action control | 0.0% | 100.0% |
| Checkpoint before pause | 0.0% | 100.0% |
| Trace completeness | 47.2% | 100.0% |
| Duplicate side effects | 0 | 0 |

## Changed cases

- Improvements: approval-pause, approval-resume, step-budget, tool-timeout, verification-failure
- Regressions: none

## Release gate

- [x] `all_candidate_cases_pass`
- [x] `at_least_one_measured_improvement`
- [x] `no_case_regressions`
- [x] `approval_pauses_before_write`
- [x] `approved_resume_writes_once`
- [x] `timeout_is_explicit`
- [x] `step_budget_is_enforced`
- [x] `verification_failure_is_not_success`
