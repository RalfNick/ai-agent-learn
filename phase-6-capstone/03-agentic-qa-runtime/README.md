# Phase6 03 Agentic QA Runtime

这一节把 Phase6 从“知识库可检索”推进到“后端可以基于知识库回答”。

当前 runtime 不调用 LLM，但已经使用 LangGraph `StateGraph` 表达 Agentic QA 控制流：

```text
request.received
  ↓
retrieve
  ↓
context_grade
  ├── enough evidence → answer.generate → evidence_review
  │       ├── supported → END
  │       └── failed → answer.repair → evidence_review
  └── weak evidence   → abstain
```

为什么先做 LLM-free 版本？

- API contract 可以稳定测试。
- sources / trace / review_status 可以先接通。
- abstain 行为可以独立验证。
- 后续接 LLM 时，只替换节点内部的 answer/review 实现，不破坏后端接口。

## 目录结构

```text
03-agentic-qa-runtime/
├── agentic_qa/
│   ├── models.py      # QAResponse / QASource / QATraceStep
│   ├── evidence.py    # evidence cleaning / review / deterministic answer
│   ├── workflow.py    # LangGraph StateGraph workflow
│   └── runtime.py     # AgenticQARuntime
├── run_agentic_qa.py  # CLI smoke
└── tests/
```

## 运行测试

从仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/03-agentic-qa-runtime/tests
```

## CLI Smoke

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase-6-capstone/03-agentic-qa-runtime/run_agentic_qa.py \
  --source docs/phase-6 \
  --question "Phase6 为什么需要 trace？"
```

输出会包含：

- `mode=agentic_rag`
- `sources`
- `trace`
- `review_status`
- `context_score`

## 和后端的关系

`01-backend-skeleton` 现在支持 runtime 注入：

```python
create_app(runtime=...)
```

默认仍然使用 placeholder runtime，不影响第一阶段的独立运行。

测试里通过 adapter 把 `AgenticQARuntime` 接入 `/api/v1/answer`，验证返回结构仍然符合 `AnswerResponse`：

```text
question
session_id
answer
mode
sources
trace
review_status
```

## 当前边界

当前 answer generation 会过滤 Markdown 图片、代码块、命令行和疑问句，并把有效表格行转换成自然语言证据。例如：

```text
| trace 展示 | 开发者要能调试路径 |
```

会进入回答为：

```text
trace 展示：开发者要能调试路径
```

这一节还没有做：

- LLM answer generation
- query rewrite
- faithfulness judge
- 流式输出

这些会在后续升级。当前 slice 已经证明：Capstone 的 API 可以接入一个可检索、可拒答、可追踪，并且具备 repair 路由的 LangGraph Agentic QA runtime。
