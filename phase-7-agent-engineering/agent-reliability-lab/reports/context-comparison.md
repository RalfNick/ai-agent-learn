# Context Architecture Comparison

- Context eval version: `0.3.0`
- Release gate: **PASS**

| Metric | dump-all-v1 | context-packet-v1 |
| --- | ---: | ---: |
| Case pass rate | 37.5% | 100.0% |
| Required topic coverage | 87.5% | 100.0% |
| Invalid-source cases | 3 | 0 |
| Irrelevant-source cases | 3 | 0 |
| Budget compliance | 100.0% | 100.0% |
| Missing-evidence accuracy | 75.0% | 100.0% |
| Average estimated tokens | 66.38 | 50.25 |

Estimated tokens come from the lab's deterministic estimator, not a provider tokenizer.

## Changed cases

- Improvements: ctx-001, ctx-002, ctx-005, ctx-006, ctx-008
- Regressions: none

## Release gate

- [x] `candidate_not_worse_overall`
- [x] `at_least_one_measured_improvement`
- [x] `no_case_regressions`
- [x] `all_required_available_evidence_retained`
- [x] `no_forbidden_source_exposure`
- [x] `no_irrelevant_source_exposure`
- [x] `all_packets_within_budget`
- [x] `missing_evidence_is_explicit`
