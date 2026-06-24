# Agent API 回归测试：别只测接口通不通

![Phase5 Testing Eval Architecture](diagram/testing-eval/testing-eval-architecture.svg)

到了 Phase5，Agent 服务已经经历了三步：

```text
FastAPI 包装：脚本变成 HTTP 服务
Docker 部署：服务有可重复启动方式
Observability：请求和 Agent runtime 有 trace
```

最后还差一个问题：以后改代码时，怎么知道这个 Agent 服务没有悄悄退化？

普通 API 可以测：

```text
status_code == 200
response schema 正确
```

但 Agent API 只测这些远远不够。因为一次 Agent 响应可能形式正确，但实际已经坏了：

```text
没有调用工具
没有证据
review 没过
runtime trace 路径不对
成本或延迟异常
```

所以 Phase5 的第四步做了一个最小评估入口：用固定样本 replay 当前 Agent runtime，并检查关键质量条件。

对应代码在：

- `phase-5-production/01-fastapi-backend/app/evaluation.py`
- `phase-5-production/01-fastapi-backend/app/main.py`
- `phase-5-production/01-fastapi-backend/app/schemas.py`
- `phase-5-production/01-fastapi-backend/tests/test_api.py`
- `phase-5-production/04-testing-eval/README.md`

## 一、回归测试和评估不是一回事

这一步先区分两个概念。

| 类型 | 主要回答 | 示例 |
| --- | --- | --- |
| API 回归测试 | 接口是否仍然按约定工作 | endpoint 存在、schema 正确、错误码正确 |
| Agent 评估 | 这次回答质量是否达标 | 是否有证据、是否调用工具、review 是否通过 |
| 线上 replay | 历史真实问题是否还能稳定回答 | 把线上样本重新跑一遍，看通过率 |

当前工程做的是最小版：

```text
固定样本集 + 确定性 judge + API 触发 replay + observability trace
```

它不是 RAGAS，也不是完整 LLM-as-judge。但它已经能证明一件事：Agent 服务开始具备“可回归”的能力。

## 二、评估样本长什么样

评估样本定义在 `app/evaluation.py`。

每条 case 不是只包含问题，还包含验收条件：

```python
@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_review_status: str
    required_trace_steps: list[str]
    minimum_evidence_count: int
    required_tool_names: list[str]
```

例如：

```python
EvaluationCase(
    case_id="phase4-memory-evidence",
    question="请结合 Phase4 Memory 的代码、文章和测试证据，说明是否可以进入 Phase5",
    expected_review_status="approved",
    required_trace_steps=["runtime.start", "memory.search", "reviewer.review"],
    minimum_evidence_count=3,
    required_tool_names=["search_docs", "find_code_examples", "read_benchmark_summary"],
)
```

这比“预期答案等于某段文本”更适合 Agent 系统。

因为 Agent 的回答文本可能每次略有不同，但工程质量条件应该稳定：

```text
该查文档时查文档
该查代码时查代码
该看 benchmark 时看 benchmark
最终 review 通过
证据数量达标
trace 关键节点存在
```

## 三、EvaluationRunner 做了什么

核心类是 `EvaluationRunner`。

它负责三件事：

```text
1. 选择要运行的 case
2. 调用 RuntimeAdapter.answer()
3. 用确定性规则判断是否通过
```

调用路径是：

```text
POST /api/v1/evaluations/run
        ↓
EvaluationRunner.run()
        ↓
RuntimeAdapter.answer()
        ↓
Phase4 IntegratedAgentRuntime
        ↓
evaluate_answer()
        ↓
pass_rate + per-case result
```

这里最重要的设计选择是：评估没有绕过 API runtime。

它复用的是同一个 `RuntimeAdapter`，也就是和 `/api/v1/agent/answer` 一样的执行路径。这样评估才接近真实服务，而不是另写一个离线脚本自嗨。

## 四、judge 不是模型，而是工程验收规则

当前的 judge 是确定性的：

```python
def evaluate_answer(case: EvaluationCase, answer: AnswerResponse) -> list[str]:
    failures: list[str] = []
    tool_names = {item.tool_name for item in answer.tool_results}
    trace_steps = set(answer.trace)

    if answer.review.status != case.expected_review_status:
        failures.append(...)

    if len(answer.evidence) < case.minimum_evidence_count:
        failures.append(...)

    for required_tool in case.required_tool_names:
        if required_tool not in tool_names:
            failures.append(...)

    for required_step in case.required_trace_steps:
        if required_step not in trace_steps:
            failures.append(...)

    return failures
```

这段规则朴素，但很实用。

它检查的不是“文字像不像标准答案”，而是 Agent 工作流有没有按预期发生。

| 检查项 | 意义 |
| --- | --- |
| `review_status` | 最终质量门是否通过 |
| `evidence_count` | 回答是否有足够证据 |
| `required_tool_names` | 是否真的调用关键工具 |
| `required_trace_steps` | runtime 是否走过关键节点 |

这和 Phase3 的 Agentic RAG benchmark 一脉相承：不要只相信答案，要看证据和执行路径。

## 五、为什么评估结果也要写入 observability

`EvaluationRunner` 运行每条 case 后，会写一条 `AgentRunObservation`。

trace id 规则是：

```text
eval-{case_id}
```

例如：

```text
eval-phase4-memory-evidence
```

这样运行评估后，可以继续查：

```bash
curl http://127.0.0.1:8000/api/v1/observability/traces/eval-phase4-memory-evidence
```

这件事很关键。

如果某条 eval case 失败，不应该只看到：

```json
{"passed": false}
```

还要能继续追：

```text
这次 runtime trace 是什么？
调用了哪些工具？
证据数量是多少？
review 状态是什么？
估算成本是多少？
```

评估和可观测性接上以后，失败才有调试入口。

## 六、新增的两个接口

查看内置 case：

```text
GET /api/v1/evaluations/cases
```

运行评估：

```text
POST /api/v1/evaluations/run
```

请求体：

```json
{
  "case_ids": ["phase4-memory-evidence"],
  "session_prefix": "manual-eval"
}
```

`case_ids` 可以不传，不传就跑全部内置样本。

返回结构：

```json
{
  "total_cases": 3,
  "passed_cases": 3,
  "failed_cases": 0,
  "pass_rate": 1.0,
  "average_latency_ms": 112.37,
  "estimated_cost_usd": 0.00016855,
  "results": [
    {
      "case_id": "phase4-memory-evidence",
      "trace_id": "eval-phase4-memory-evidence",
      "passed": true,
      "failures": [],
      "review_status": "approved",
      "evidence_count": 12,
      "tool_names": ["search_docs", "find_code_examples", "read_benchmark_summary"]
    }
  ]
}
```

这组数字来自当前工程的 smoke run：

```text
total_cases=3
passed_cases=3
failed_cases=0
pass_rate=1.0
average_latency_ms=112.37
estimated_cost_usd=0.00016855
```

这里的数字不是要证明系统“很强”，而是证明评估链路已经能输出可复盘结果。

## 七、测试如何覆盖

`tests/test_api.py` 里新增了两类测试。

第一类，确认样本接口存在：

```python
response = self.client.get("/api/v1/evaluations/cases")

self.assertEqual(response.status_code, 200)
body = response.json()
self.assertGreaterEqual(len(body["cases"]), 3)
```

第二类，确认评估能真实 replay 并通过：

```python
response = self.client.post(
    "/api/v1/evaluations/run",
    json={
        "case_ids": ["phase4-memory-evidence"],
        "session_prefix": "eval-test",
    },
)

body = response.json()
self.assertEqual(body["total_cases"], 1)
self.assertEqual(body["passed_cases"], 1)
self.assertEqual(body["pass_rate"], 1.0)
```

第三类，确认 eval trace 能被 observability 查到：

```python
trace_response = self.client.get(
    f"/api/v1/observability/traces/{result['trace_id']}"
)

self.assertEqual(trace_response.status_code, 200)
self.assertEqual(trace["agent"]["review_status"], "approved")
self.assertGreaterEqual(trace["agent"]["tool_count"], 3)
```

这组测试把 Phase5 的几个部分串起来了：

```text
API endpoint
RuntimeAdapter
EvaluationRunner
ObservabilityStore
Trace detail
```

## 八、怎么运行

启动服务：

```bash
cd phase-5-production/01-fastapi-backend
python3 -m uvicorn app.main:app --reload --port 8000
```

查看样本：

```bash
curl http://127.0.0.1:8000/api/v1/evaluations/cases
```

运行全部样本：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/evaluations/run \
  -H "Content-Type: application/json" \
  -d '{"session_prefix":"manual-eval"}'
```

运行单条样本：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/evaluations/run \
  -H "Content-Type: application/json" \
  -d '{"case_ids":["phase4-memory-evidence"],"session_prefix":"manual-eval"}'
```

查看 eval trace：

```bash
curl http://127.0.0.1:8000/api/v1/observability/traces/eval-phase4-memory-evidence
```

运行测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-5-production/01-fastapi-backend/tests
```

## 九、当前边界

当前评估还有明显边界：

| 当前实现 | 后续升级 |
| --- | --- |
| 内置 3 条样本 | 接入真实线上问题和黄金集 |
| 确定性 judge | 增加 LLM-as-judge 或 RAGAS |
| 只看工具和 trace | 增加答案语义正确性 |
| 内存观测 | 接入持久化 trace 后端 |
| 手动触发 | 接入 CI 或 nightly replay |

但这一步已经足够作为 Phase5 的收口。

因为现在这个服务不只是能跑、能部署、能观测，还能被回归验证。

## 十、Phase5 到这里证明了什么

Phase5 的主线已经完整：

```text
01 FastAPI：服务边界
02 Docker：部署边界
03 Observability：调试边界
04 Testing Eval：质量边界
```

这四步合在一起，才叫“生产化入门”。

下一阶段进入 Phase6 capstone 时，就不是从零搭一个大系统了，而是可以把这些能力组合起来：

```text
企业知识库 Agent
FastAPI 服务
可部署容器
trace 和观测
回归评估
前端交互
```

Agent 工程真正麻烦的地方，不是写出一次看起来聪明的回答，而是让系统在不断修改后仍然可解释、可验证、可恢复。
