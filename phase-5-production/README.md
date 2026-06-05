# Phase5 Production

Phase5 的目标是把前面已经能跑的 Agent runtime 逐步变成可部署、可观测、可测试的服务。

当前建议顺序：

```text
phase-5-production/
├── 01-fastapi-backend/   # 把 Phase4 runtime 包成 HTTP API
├── 02-docker-deploy/     # Dockerfile + Compose + healthcheck
├── 03-observability/     # trace、日志、成本和延迟观测
└── 04-testing-eval/      # API 回归测试和线上评估入口
```

## 当前状态

- `01-fastapi-backend/`：已完成最小 FastAPI 服务，包含 `/health` 和 `/api/v1/agent/answer`。

## 学习重点

Phase5 不再重新设计 Agent 能力，而是围绕生产边界展开：

```text
服务接口是否稳定？
错误能不能被定位？
部署是否可重复？
调用链路是否可观测？
测试能不能覆盖核心行为？
```

第一阶段先把服务边界立住，再进入 Docker 和观测。
