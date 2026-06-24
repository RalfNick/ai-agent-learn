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
- `02-docker-deploy/`：已完成 Dockerfile、Compose、本地持久化 volume 和 `/health` healthcheck。
- `03-observability/`：已完成 trace id、HTTP/Agent latency、runtime trace、review、证据数量和估算成本观测。
- `04-testing-eval/`：已完成内置 eval cases、API replay、确定性 judge、pass rate 和 eval trace。

## 学习重点

Phase5 不再重新设计 Agent 能力，而是围绕生产边界展开：

```text
服务接口是否稳定？
错误能不能被定位？
部署是否可重复？
调用链路是否可观测？
测试能不能覆盖核心行为？
```

当前已经完成服务边界、Docker 本地部署闭环、最小观测层和 API 回归评估入口。Phase5 可以开始收口，下一步进入 Phase6 capstone。
