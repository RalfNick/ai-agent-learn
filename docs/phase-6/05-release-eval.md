# Phase6 第五块：给 Capstone 一个验收闭环

前四块已经完成：

```text
Backend skeleton
Knowledge ingestion
LangGraph Agentic QA runtime
Web UI
```

第五块 `05-release-eval` 负责收口：让这个系统不只是“能跑”，而是能被验收。

对应代码：

- `phase-6-capstone/05-release-eval/release_eval/evaluator.py`
- `phase-6-capstone/05-release-eval/run_eval.py`
- `phase-6-capstone/05-release-eval/api_server.py`
- `phase-6-capstone/05-release-eval/eval_cases.json`
- `phase-6-capstone/05-release-eval/docker-compose.yml`

## 一、为什么需要 release eval

Agent 项目很容易出现一种错觉：

```text
我试了一个问题，回答看起来还行，所以系统完成了。
```

这不够。

Phase6 至少要能回答：

```text
哪些问题是固定验收样本？
每个样本为什么算通过？
回答有没有 sources？
review_status 是否符合预期？
改完代码后是否退化？
```

所以这里做了一个很小的 golden set eval。

## 二、Eval 如何判断通过

`EvalCase` 包含：

```python
case_id
question
expected_terms
expected_review_status
expected_source_title
forbidden_terms
expected_trace_steps
min_context_score
top_k
force_unsafe_answer
```

当前判断仍然保持轻量，但不只看“答案里有没有几个词”：

```text
review_status 必须符合预期。
answer 必须包含 expected terms。
answer 不能包含 forbidden terms。
sources 必须包含 expected source title。
trace 必须按顺序包含 expected trace steps。
case 可以单独调高 min_context_score，验证 abstain。
case 可以强制 unsafe answer builder，验证 review.failed → answer.repair。
```

它不是 RAGAS，也不是完整 benchmark。

它的价值是做 release smoke：至少知道核心链路、拒答路径、修复路径和来源约束有没有断。

当前 `eval_cases.json` 有 5 个样本：

| case | 目的 |
| --- | --- |
| `trace-value` | 正常回答，验证 trace 价值和来源 |
| `web-ui-observability` | 正常回答，验证 UI 可观测字段 |
| `unrelated-policy-abstain` | 领域外问题，必须 abstain |
| `weak-retrieval-abstain` | 弱检索问题，必须 abstain |
| `repair-removes-unsupported-claim` | 注入无来源结论，必须 repair 并移除禁止词 |

## 三、Integrated API

前面 `01-backend-skeleton` 默认还是 placeholder。

本节新增 `api_server.py`，把它接到真实 runtime：

```text
create_app(runtime=AgenticRuntimeAdapter(...))
```

也就是说，release 入口不再返回 placeholder，而是：

```text
FastAPI
  ↓
LangGraph Agentic QA runtime
  ↓
Knowledge index
  ↓
answer + sources + trace + review_status
```

## 四、Compose 的定位

`docker-compose.yml` 定义两个服务：

```text
backend: Python + FastAPI + Agentic runtime
web: Node + Next.js UI
```

这不是最终生产镜像。

它使用官方镜像加 volume 挂载，目的是学习阶段快速复现：

```bash
cd phase-6-capstone/05-release-eval
docker compose up
```

后续如果要做真正生产化，可以再拆 Dockerfile、多阶段构建、缓存依赖和健康检查。

## 五、本轮 review

这一轮达到了预期：

```text
golden set eval 可以跑。
5 个 Phase6 smoke case 当前全部通过。
FastAPI placeholder 有了 integrated API 入口。
Web UI 和 backend 有 compose 串联方式。
eval 能检查 abstain / repair / trace route / forbidden terms / wrong source。
```

当前仍然没做：

```text
真实 LLM judge
更大规模 golden set
CI 自动运行 eval
生产镜像优化
权限和登录
```

到这里，Phase6 学习版 capstone 已经形成了完整闭环：能导入资料、检索、LangGraph 路由、回答、展示、验收。
