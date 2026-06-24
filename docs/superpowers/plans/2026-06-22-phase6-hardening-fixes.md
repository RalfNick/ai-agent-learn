# Phase6 hardening fixes

## Goal

Tighten the Phase6 capstone integration after the system-design review:

1. Add backend CORS support for the local Web UI and cover it with an API test.
2. Make Web UI API failures explicit instead of showing the demo answer as if it answered the current question.
3. Expand release evaluation beyond happy-path assertions with abstain, repair, weak retrieval, wrong-source, forbidden-term, and trace-route checks.

## Red tests

- Backend: preflight `OPTIONS /api/v1/answer` from `http://127.0.0.1:3020` must return CORS headers.
- Web UI: fallback response must be an explicit client error response with empty sources and client-side trace, not a copied demo answer.
- Release eval: evaluator must support per-case retrieval thresholds, expected trace steps, forbidden answer terms, repair forcing, and wrong-source failure detection.

## Implementation

- Add configurable `allowed_origins` to backend settings and install `CORSMiddleware` inside `create_app`.
- Add `buildFallbackAnswerResponse()` in the Web UI lib layer and use it from `app/page.tsx` catch handling.
- Extend `EvalCase` and `EvalRecord` with richer checks, building a runtime per case when case-level runtime behavior is required.
- Add more representative `eval_cases.json` cases while keeping release eval expected to pass.

## Verification

- `python3 -m unittest discover -s phase-6-capstone/01-backend-skeleton/tests`
- `npm test --prefix phase-6-capstone/04-web-ui`
- `npm run build --prefix phase-6-capstone/04-web-ui`
- `python3 -m unittest discover -s phase-6-capstone/05-release-eval/tests`
- `python3 phase-6-capstone/05-release-eval/run_eval.py --source docs/phase-6 --cases phase-6-capstone/05-release-eval/eval_cases.json`
