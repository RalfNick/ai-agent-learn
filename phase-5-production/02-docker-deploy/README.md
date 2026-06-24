# Phase5 02 Docker Deploy

这一节把 `phase-5-production/01-fastapi-backend/` 的 FastAPI 服务封装成一个可重复启动的容器。镜像里会复制两部分代码：

- `phase-5-production/01-fastapi-backend/app/`：HTTP API 层。
- `phase-4-advanced/03-memory-system/`、`04-multi-agent-patterns/`、`05-agent-runtime-integration/`：Phase4 runtime 依赖。
- `docs/`、Phase2/Phase3 benchmark outputs：Phase4 runtime 的只读项目工具需要这些资料作为 evidence 来源。

## 启动

先确认 Docker Desktop 或 Docker daemon 已启动：

```bash
docker --version
docker compose version
docker compose config
```

从本目录运行：

```bash
docker compose up --build
```

或者从仓库根目录运行：

```bash
docker compose -f phase-5-production/02-docker-deploy/docker-compose.yml up --build
```

服务启动后访问：

```bash
curl http://127.0.0.1:8000/health
```

预期返回：

```json
{"status":"ok","service":"phase5-agent-api","phase":"phase-5","version":"0.1.0"}
```

## 调用 Agent API

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"Phase4 当前 runtime 集成了哪些能力？","session_id":"docker-demo"}'
```

## 健康检查

健康检查在两层配置：

- Dockerfile：`HEALTHCHECK` 直接访问 `http://127.0.0.1:8000/health`。
- Compose：`healthcheck` 使用同一个 `/health` 端点，便于本地观察服务状态。

查看状态：

```bash
docker compose ps
```

## 持久化

长期记忆写入容器内：

```text
/app/phase-5-production/01-fastapi-backend/.memory
```

Compose 使用 named volume `phase5_agent_memory` 持久化这个目录，避免容器重建后会话记忆丢失。

## 停止

```bash
docker compose down
```

如果需要同时清理本地记忆 volume：

```bash
docker compose down -v
```
