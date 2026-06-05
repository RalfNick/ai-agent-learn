# Phase5 FastAPI Backend

本目录是 Phase5 的第一步：把 Phase4 的 `IntegratedAgentRuntime` 包成一个可测试的 FastAPI 服务。

这一阶段先不做 Docker、Langfuse、鉴权和流式输出。先把最小服务边界跑通：

```text
HTTP Request
  -> Pydantic request schema
  -> RuntimeAdapter
  -> Phase4 IntegratedAgentRuntime
  -> Pydantic response schema
  -> HTTP Response
```

## 文件结构

```text
phase-5-production/01-fastapi-backend/
├── app/
│   ├── config.py             # 服务配置和路径
│   ├── main.py               # FastAPI app factory 和路由
│   ├── runtime_adapter.py    # Phase4 runtime 到 API response 的转换
│   └── schemas.py            # 请求/响应 schema
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

## 安装依赖

```bash
python3 -m pip install -r phase-5-production/01-fastapi-backend/requirements.txt
```

## 运行测试

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/01-fastapi-backend/tests
```

## 启动服务

```bash
cd phase-5-production/01-fastapi-backend
uvicorn app.main:app --reload --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

调用 Agent：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/answer \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "请结合 Phase4 Memory 的代码、文章和测试证据，说明是否可以进入 Phase5",
    "session_id": "demo"
  }'
```

## 当前 API

| Method | Path | 作用 |
|--------|------|------|
| `GET` | `/health` | 返回服务元数据 |
| `POST` | `/api/v1/agent/answer` | 调用 Phase4 runtime，返回 answer、evidence、review、trace |

## 当前边界

第一版只做服务化入口，不做：

```text
真实 LLM 调用
SSE / WebSocket 流式输出
鉴权
限流
Docker
Langfuse
线上错误追踪
```

这些会在 Phase5 后续阶段继续补。当前目标是先把 Agent runtime 变成一个明确、可测试、可扩展的 HTTP 服务。
