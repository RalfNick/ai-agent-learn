# Phase6 05 Release Eval

这一节是 Phase6 capstone 的收口集成。

它做三件事：

- golden set eval：用固定问题检查 answer、sources、review_status、trace route 和禁止词。
- integrated API entry：把 FastAPI skeleton 和 LangGraph QA runtime 接起来。
- compose 配置：描述 backend + web 的本地演示启动方式。

## Eval

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase-6-capstone/05-release-eval/run_eval.py \
  --source docs/phase-6 \
  --cases phase-6-capstone/05-release-eval/eval_cases.json
```

当前 eval cases：

- `trace-value`
- `web-ui-observability`
- `unrelated-policy-abstain`
- `weak-retrieval-abstain`
- `repair-removes-unsupported-claim`

通过标准：

- `review_status` 符合预期。
- answer 包含 expected terms。
- answer 不包含 forbidden terms。
- sources 包含 expected source title。
- trace 按顺序包含 expected trace steps。
- case 可以单独设置 `min_context_score` / `top_k`。
- case 可以设置 `force_unsafe_answer` 来验证 `review.failed -> answer.repair`。

## Integrated API

本节提供：

```text
phase-6-capstone/05-release-eval/api_server.py
```

它把：

```text
01-backend-skeleton/create_app
03-agentic-qa-runtime/build_runtime_from_sources
```

接成一个真实 `/api/v1/answer` 服务。

本地运行：

```bash
cd phase-6-capstone/05-release-eval
uvicorn api_server:app --host 0.0.0.0 --port 8010
```

## Compose

```bash
cd phase-6-capstone/05-release-eval
docker compose config
docker compose up
```

端口：

- backend: `http://127.0.0.1:8010`
- web: `http://127.0.0.1:3020`

## 当前边界

- compose 使用官方 Python/Node 镜像和 volume 挂载，方便学习阶段快速验证。
- 还没有生产镜像分层优化。
- 还没有真实 LLM faithfulness judge。
- 还没有 CI pipeline。

这已经足够作为 Phase6 学习版 capstone 的本地 release/eval 闭环。
