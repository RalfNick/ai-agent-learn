# Phase4 Advanced Agent Systems

Phase4 的目标不是继续增加框架 demo，而是补齐企业级 Agent 的进阶能力：工具协议、安全边界、长期状态、多 Agent 协作和后续生产化入口。

Phase1 已经手写 Agent 核心循环，Phase2 已经完成 RAG benchmark，Phase3 已经用 LangGraph 做了 Agentic RAG。Phase4 要回答的问题是：

```text
Agent 如何安全地连接真实工具？
Agent 如何管理权限和边界？
Agent 如何跨会话保留状态？
多个 Agent 如何协作但不失控？
```

## 学习顺序

当前建议顺序：

```text
phase-4-advanced/
├── 01-mcp-server/          # 第一优先级：工具协议和资源暴露
├── 02-agent-security/      # Guardrails、权限、Prompt 注入防护
├── 03-memory-system/       # 长期状态、实体记忆、跨会话恢复
└── 04-multi-agent-patterns/# Supervisor、handoff、协作协议
```

这个顺序和原计划略有调整：MCP 提前，Memory 后移。

原因是 Phase2 已经做过 memory-rag，继续先做记忆容易重复检索增强；MCP 更适合作为 Phase4 入口，因为它能自然引出工具边界、安全授权和生产集成。

## 第一个实战：ai-agent-learn MCP Server

第一版建议做一个当前学习工程自己的 MCP Server。

目标不是把功能做大，而是把当前工程变成 Agent 可以安全访问的知识库和工具集合。

建议最小工具集：

| 工具 | 作用 |
|------|------|
| `search_docs` | 搜索 `docs/` 下的学习文章 |
| `find_code_examples` | 按关键词查找各 phase 的示例代码 |
| `read_benchmark_summary` | 读取 Phase2 / Phase3 benchmark 汇总结果 |

建议先暴露只读能力。

第一版不做：

```text
不写文件
不跑 benchmark
不执行 shell
不修改文章
不访问工程外目录
```

这些限制不是偷懒，而是为了先把 MCP 的资源、工具、权限边界学清楚。

## 验收标准

Phase4 第一个 MCP Server 完成时，至少满足：

- 能通过 MCP 客户端列出工具。
- `search_docs` 能返回文章路径、标题和匹配片段。
- `find_code_examples` 能返回脚本路径和所属 phase。
- `read_benchmark_summary` 能返回 Phase2 / Phase3 的关键指标。
- 所有工具默认只读，不越过当前工程目录。
- 有 smoke test 或最小客户端调用示例。

## 文章输出

建议文章标题：

```text
MCP 实战：把学习工程变成 Agent 可调用的工具服务
```

文章不要只介绍协议概念，而要回答：

```text
为什么 Agent 需要 MCP？
Tool、Resource、Prompt 分别解决什么？
为什么第一版 MCP Server 只做只读？
权限边界如何设计？
MCP 和 LangGraph / Agentic RAG 如何衔接？
```

## 当前状态

Phase4 尚未正式开始。

进入 Phase4 前，先阅读：

```text
docs/phase-3/phase3-review-and-next-steps.md
```

再进入：

```text
phase-4-advanced/01-mcp-server/
```
