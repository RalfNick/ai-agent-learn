# Phase5 03 Observability

这一节给 `01-fastapi-backend` 增加最小可观测性能力。实现代码仍放在 FastAPI 服务里，因为观测点必须贴着真实请求路径产生。

## 已实现能力

- `X-Trace-Id`：每个请求都会返回 trace id，也可以由调用方传入。
- HTTP 指标：method、path、status code、latency。
- Agent 指标：runtime trace、tool count、evidence count、review status、estimated cost。
- 查询接口：
  - `GET /api/v1/observability/summary`
  - `GET /api/v1/observability/traces/{trace_id}`

## 运行

```bash
cd phase-5-production/01-fastapi-backend
python3 -m uvicorn app.main:app --reload --port 8000
```

调用 Agent API：

```bash
curl -i -X POST http://127.0.0.1:8000/api/v1/agent/answer \
  -H "Content-Type: application/json" \
  -H "X-Trace-Id: phase5-observe-demo" \
  -d '{"question":"请结合 Phase4 Memory 的代码和测试证据，说明当前状态","session_id":"observe-demo"}'
```

查看 summary：

```bash
curl http://127.0.0.1:8000/api/v1/observability/summary
```

查看单条 trace：

```bash
curl http://127.0.0.1:8000/api/v1/observability/traces/phase5-observe-demo
```

## 验收

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/01-fastapi-backend/tests
```

当前实现是内存版观测，不负责跨进程持久化。后续可以把 `ObservabilityStore` 替换为 OpenTelemetry、Langfuse、Prometheus 或数据库。
