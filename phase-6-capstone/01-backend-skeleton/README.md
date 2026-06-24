# Phase6 01 Backend Skeleton

这是 Phase6 capstone 的第一个可运行切片：企业知识库 Agent 的 FastAPI 后端骨架。

这一节不接真实 RAG，也不接 LangGraph。目标是先把服务边界固定下来：

- `GET /health`
- `POST /api/v1/answer`
- `GET /api/v1/observability/summary`

`/api/v1/answer` 当前返回 `mode=placeholder`，并保留后续 UI 和 runtime 会用到的字段：

- `answer`
- `sources`
- `trace`
- `review_status`
- `session_id`

## 安装

```bash
cd phase-6-capstone/01-backend-skeleton
python3 -m pip install -r requirements.txt
```

## 运行测试

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/01-backend-skeleton/tests
```

## 启动服务

```bash
cd phase-6-capstone/01-backend-skeleton
python3 -m uvicorn app.main:app --reload --port 8010
```

## 调用

健康检查：

```bash
curl http://127.0.0.1:8010/health
```

问答接口：

```bash
curl -X POST http://127.0.0.1:8010/api/v1/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"Phase6 capstone 要做什么？","session_id":"demo"}'
```

观测摘要：

```bash
curl http://127.0.0.1:8010/api/v1/observability/summary
```

## 当前边界

当前回答是 placeholder，不做检索、不调用模型、不产生真实来源。后续阶段会替换 runtime：

- `02-knowledge-ingestion`：接入文档导入、chunk、index 和 retrieval。
- `03-agentic-qa-runtime`：接入 LangGraph Agentic RAG。
- `04-web-ui`：展示回答、sources 和 trace。
