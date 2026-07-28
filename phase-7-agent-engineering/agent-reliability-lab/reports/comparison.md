# Agent Eval Comparison

- Eval version: `0.2.0`
- Trials per task: `3`
- Release gate: **PASS**

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Trial pass rate | 60.0% | 95.0% |
| Task pass rate | 60.0% | 95.0% |
| Correct abstention rate | 100.0% | 100.0% |
| False answer rate | 0.0% | 0.0% |
| Stability rate | 100.0% | 100.0% |
| Median latency (local ms) | 0.036 | 0.034 |
| p95 latency (local ms) | 0.058 | 0.043 |

Local latency is diagnostic only and is not part of the release gate.

## Changed tasks

- Improvements: cap-001, cap-002, cap-003, cap-004, cap-005, cap-006, cap-007
- Regressions: none
- Unstable candidate tasks: none

## Release gate

- [x] `candidate_not_worse_overall`
- [x] `at_least_one_measured_improvement`
- [x] `no_task_regressions`
- [x] `no_safety_regressions`
- [x] `no_false_answer_increase`
- [x] `candidate_trials_are_stable`
