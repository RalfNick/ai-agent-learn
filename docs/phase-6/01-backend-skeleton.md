# Phase6 第一块：先把 Agent 服务边界立起来

Phase6 的目标很大：做一个企业知识库 Agent。

但第一步不能上来就写 LangGraph、向量库、前端和 Docker Compose。系统越大，越应该先确认一件事：

```text
用户、前端、Agent runtime、观测和评估，最后到底通过什么接口协作？
```

所以 Phase6 的第一个实现切片是 `01-backend-skeleton`。

对应代码在：

- `phase-6-capstone/01-backend-skeleton/app/main.py`
- `phase-6-capstone/01-backend-skeleton/app/schemas.py`
- `phase-6-capstone/01-backend-skeleton/app/runtime.py`
- `phase-6-capstone/01-backend-skeleton/app/observability.py`
- `phase-6-capstone/01-backend-skeleton/app/config.py`
- `phase-6-capstone/01-backend-skeleton/tests/test_api.py`

## 一、为什么不是先写 RAG

Phase2 已经证明了 RAG pipeline。

Phase3 已经证明了 Agentic RAG。

Phase5 已经证明了 FastAPI 服务化。

但 Capstone 是组合工程，真正容易乱的是边界：

```text
前端需要什么字段？
后端返回 sources 的结构是什么？
trace 怎么展示？
review_status 放在哪里？
placeholder 阶段和真实 runtime 阶段如何兼容？
```

如果这些接口不先定住，后面每接一个模块都会改一遍 API，前端、测试和评估也会跟着抖。

所以第一步只做服务骨架，不做真实智能。

## 二、当前接口

当前有三个接口：

```text
GET  /health
POST /api/v1/answer
GET  /api/v1/observability/summary
```

`/health` 用来确认服务启动：

```json
{
  "status": "ok",
  "service": "phase6-capstone-api",
  "phase": "phase-6",
  "version": "0.1.0"
}
```

`/api/v1/answer` 当前返回 placeholder：

```json
{
  "question": "Phase6 capstone 要做什么？",
  "session_id": "demo-session",
  "answer": "Phase6 backend skeleton is ready...",
  "mode": "placeholder",
  "sources": [],
  "trace": [
    {"step": "request.received", "detail": "Accepted answer request."},
    {"step": "runtime.placeholder", "detail": "Returned skeleton response without retrieval."},
    {"step": "response.placeholder", "detail": "Sources and review are intentionally empty."}
  ],
  "review_status": null
}
```

这里最重要的不是答案内容，而是字段结构。

`sources` 和 `trace` 现在是空数据和占位路径，但字段已经存在。后面接入真实 RAG 和 LangGraph 时，可以替换 runtime，不需要改前端接口。

还有一个容易漏掉的工程细节：Web UI 和 API 是两个本地端口。

```text
Web UI: http://127.0.0.1:3020
API:    http://127.0.0.1:8010
```

浏览器从 `3020` 调 `8010`，会触发 CORS。`app/config.py` 里把本地 UI origin 放进 allowlist，`app/main.py` 通过 `CORSMiddleware` 显式开放 `GET / POST / OPTIONS`。

这不是“生产安全配置”，但它让 Phase6 的本地联调路径变成真实浏览器路径，而不是只靠 curl。

## 三、schemas 先行

`app/schemas.py` 定义了当前 API 合约：

```python
class AnswerResponse(BaseModel):
    question: str
    session_id: str
    answer: str
    mode: str
    sources: list[SourceItem]
    trace: list[TraceStep]
    review_status: str | None
```

这个结构提前为后续能力留了位置：

| 字段 | 当前含义 | 后续用途 |
| --- | --- | --- |
| `mode` | `placeholder` | 区分 placeholder / rag / agentic_rag |
| `sources` | 空数组 | 展示文档引用 |
| `trace` | 固定三步 | 展示 LangGraph 路由 |
| `review_status` | `null` | 展示 faithfulness/reviewer 结果 |

这就是 skeleton 的价值：先定义系统要长期稳定的接口。

## 四、runtime 是可替换的

当前 runtime 很简单：

```python
class AnswerRuntime:
    def answer(self, question: str, session_id: str) -> AnswerResponse:
        return AnswerResponse(
            question=question,
            session_id=session_id,
            mode="placeholder",
            answer="Phase6 backend skeleton is ready...",
            sources=[],
            trace=[
                TraceStep(step="request.received", detail="Accepted answer request."),
                TraceStep(step="runtime.placeholder", detail="Returned skeleton response without retrieval."),
                TraceStep(step="response.placeholder", detail="Sources and review are intentionally empty."),
            ],
            review_status=None,
        )
```

后续 `03-agentic-qa-runtime` 要替换的就是这里。

理想状态下，`main.py` 的路由不需要大改，只需要把：

```python
runtime = AnswerRuntime()
```

替换成真正的：

```python
runtime = LangGraphAgenticQARuntime(...)
```

这就是为什么现在要把 runtime 单独放进 `runtime.py`，而不是直接把逻辑写在 route 里。

## 五、观测从第一天就存在

即使当前没有真实 RAG，也保留了最小观测：

```python
class ObservabilityStore:
    total_answer_requests: int
    last_session_id: str | None
    recent_questions: Deque[str]
```

接口：

```text
GET /api/v1/observability/summary
```

返回：

```json
{
  "total_answer_requests": 1,
  "last_session_id": "summary-session",
  "recent_questions": ["Phase6 当前做到哪一步？"]
}
```

这不是完整观测系统，但它保留了一个重要习惯：Agent 服务从第一天就应该能被观察。

后面接入 LangGraph 后，这个 summary 会继续扩展 trace、latency、cost、review 等字段。

## 六、测试先定义边界

测试在 `tests/test_api.py`。

它覆盖四件事：

```text
health metadata 正确
answer 返回 placeholder contract
blank question 被拒绝
local Web UI CORS preflight 通过
observability 能记录 answer request
```

核心断言：

```python
self.assertEqual(body["mode"], "placeholder")
self.assertEqual(body["sources"], [])
self.assertEqual(body["trace"][0]["step"], "request.received")
self.assertEqual(body["trace"][-1]["step"], "response.placeholder")
self.assertIsNone(body["review_status"])
```

这些测试后面会很有用。

当我们接入真实 RAG 时，测试会提醒我们：即使 runtime 变复杂，API contract 也不能随便破坏。

## 七、怎么运行

安装依赖：

```bash
cd phase-6-capstone/01-backend-skeleton
python3 -m pip install -r requirements.txt
```

运行测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/01-backend-skeleton/tests
```

启动服务：

```bash
python3 -m uvicorn app.main:app --reload --port 8010
```

调用：

```bash
curl -X POST http://127.0.0.1:8010/api/v1/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"Phase6 capstone 要做什么？","session_id":"demo"}'
```

## 八、下一步

下一步进入 `02-knowledge-ingestion`。

它会把 `sources=[]` 变成真实文档来源：

```text
load documents
chunk
embed
index
retrieve
return SourceItem[]
```

到那一步，Phase6 才真正开始接入知识库。

但现在这一步并不空。它把服务边界、响应结构、placeholder runtime 和最小观测固定下来了。

这就是 Capstone 的第一块地基。
