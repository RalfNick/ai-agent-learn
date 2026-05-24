# Phase3 验收复盘：从框架采样到 Agentic RAG

> 复盘日期：2026-05-21  
> 当前结论：Phase3 没有偏离总目标，但需要收口。后续不再继续横向扩框架，而是进入 Phase4 的 MCP、安全、记忆和生产化能力。

---

## 一、先给结论

Phase3 原计划是学习 LangChain、LangGraph、CrewAI、Claude Agent SDK，并做框架对比。

当前实际路线变成了：

```text
框架基础采样
-> LangChain / LangGraph 基础整合
-> LangGraph 深入
-> CrewAI / Claude SDK 对比观察
-> Agentic RAG with LangGraph
-> benchmark 验证 Agent 编排价值
```

这不是偏离，而是一次合理收敛。

原因很简单：整个工程的目标不是“学完所有 Agent 框架”，而是掌握企业级 Agent 设计和开发能力。毕业项目也是企业知识库问答 Agent 系统。相比继续横向增加框架 demo，把 Phase2 的 RAG benchmark 迁移成真实 Agentic RAG，更贴近最终目标。

所以 Phase3 可以认为已经从“会跑框架 demo”进入了更重要的一层：

```text
能设计状态
能设计路由
能设计重试
能设计拒答
能设计 repair
能用 trace 和 benchmark 证明取舍
```

---

## 二、哪些内容符合原计划

原计划里 Phase3 有四个重点：LangChain、LangGraph、CrewAI、Claude Agent SDK，再加框架对比。

当前工程都有对应产物。

| 原计划 | 当前产物 | 评价 |
|------|------|------|
| LangChain 基础 | `01-framework-basics/00-langchain-foundations/`、`00-langchain-to-langgraph-foundations.md` | 已覆盖 Runnable、LCEL、Tool、Retriever 的基础心智模型 |
| LangGraph 工作流 | `01-langgraph-deep-dive/`、`02-agentic-rag-langgraph/` | 已从基础 StateGraph 进入真实 Agentic RAG |
| CrewAI 多 Agent | `02-crewai-multi-agent/` | 已用于角色协作和 hierarchical 模式观察 |
| Claude SDK / 工具循环 | `03-claude-agent-sdk/` | 已覆盖工具循环、guardrail、handoff、自主代码操作边界 |
| 框架对比 | `04-framework-showdown/`、`03-framework-comparison-after-building.md`、`04-agent-framework-landscape.md` | 已从功能 checklist 变成工程选型视角 |

这里最重要的不是“每个框架都写了多少代码”，而是已经能看出它们解决的是不同层的问题：

```text
LangChain：能力组合
LangGraph：状态和工作流编排
CrewAI：角色协作和快速原型
Claude / OpenAI 类 SDK：模型厂商原生工具循环和安全边界
```

这已经达到 Phase3 框架认知的基本要求。

---

## 三、哪些地方做了合理偏移

最大的偏移是：Phase3 后半段从“多框架均衡学习”转向了 “LangGraph + Agentic RAG 主线”。

这个偏移是合理的。

原计划里的 Phase3 是 1 周，如果四个框架都深入，很容易每个都只做成入门 demo。那样看起来覆盖面广，但对后面的 Phase6 帮助有限。

现在的做法更像这样：

```text
横向框架：知道各自边界
纵向主线：用 LangGraph 做真实系统
评价闭环：用 Phase2 benchmark 验证
```

这比单纯对比框架更接近企业 Agent 开发。

尤其是 Agentic RAG 这一段，已经不再是“演示 LangGraph 怎么 add_node”，而是在回答更真实的问题：

```text
上下文不够时，系统要不要 rewrite？
rewrite 后还不够，要不要拒答？
答案不忠实时，要不要 repair？
repair 后仍然不达标，要不要结束？
这些路径怎么记录？
这些取舍怎么用指标证明？
```

这些问题才是 Agent 工作流设计的核心。

---

## 四、Agentic RAG benchmark 结论

当前全量 benchmark 复用了 Phase2 的真实资料和 30 个标注问题。

对比对象：

```text
linear_hybrid_rerank
agentic_rag_langgraph
```

最新全量结果：

| 系统 | P@3 | R@3 | MRR | NDCG@3 | Faithfulness | 平均延迟 | 总成本 | LLM 调用 | 拒答 |
|------|-----|-----|-----|--------|--------------|----------|--------|----------|------|
| `linear_hybrid_rerank` | 0.578 | 0.436 | 0.756 | 0.511 | 0.907 | 3269ms | $0.0296 | 60 | 0 |
| `agentic_rag_langgraph` | 0.572 | 0.425 | 0.756 | 0.503 | 0.980 | 5108ms | $0.0443 | 94 | 6 |

结论不是“Agentic RAG 全面更强”。

更准确的结论是：

```text
检索指标基本持平，甚至略低。
Faithfulness 明显提升。
延迟、成本、LLM 调用数都上升。
系统更保守，出现 6 次拒答。
```

这说明 Agentic RAG 的价值主要在质量控制，而不是自动提升检索。

所以这条学习线是对的：它训练的是 Agent 工作流设计能力，而不是单纯换一个检索器。

---

## 五、Phase3 当前已经达标的能力

Phase3 现在应该能回答这些问题。

第一，LangChain 和 LangGraph 的边界是什么？

```text
LangChain 解决能力组合。
LangGraph 解决流程编排。
```

第二，什么时候用 chain，什么时候用 graph？

```text
线性、短链路、无分支：chain。
有状态、分支、循环、暂停、恢复、重试、审计：graph。
```

第三，Agentic RAG 相比 linear RAG 提升在哪里？

```text
不是检索指标，而是 Faithfulness、拒答、repair、trace 和风险控制。
```

第四，query rewrite、repair、abstain 分别解决什么问题？

```text
query rewrite：上下文不足时换一种检索表达。
repair：答案不忠实时删除无依据声明。
abstain：资料不足或风险过高时拒答。
```

第五，为什么 Agent workflow 必须有 trace？

```text
没有 trace，只能看到最终答案。
有 trace，才能知道系统在哪一步变差。
```

第六，CrewAI / Claude SDK / LangGraph 分别隐藏和暴露了什么？

```text
CrewAI 暴露角色和任务，隐藏较多调度细节。
Claude / OpenAI 类 SDK 暴露工具循环和安全边界，但编排需要自己组织。
LangGraph 暴露 State、Node、Edge、路由和 checkpoint，因此更适合可控工作流。
```

这些能力已经满足 Phase3 的核心预期。

---

## 六、当前还缺什么

Phase3 不能解决所有问题。

如果目标是企业级 Agent 开发，后面还缺四块能力：

```text
MCP 工具体系
安全边界和权限控制
长期状态与跨会话记忆
服务化、观测、部署和评估自动化
```

这也是为什么现在应该停止继续横向加框架。

继续补 AutoGen、Agno、Mastra、Vercel AI SDK，短期会增加知识面，但对当前主线帮助有限。真正该补的是后面这些生产能力。

---

## 七、Phase4 优先级调整

原计划 Phase4 的顺序是：

```text
Memory System
Multi-Agent Patterns
MCP Server
Agent Security
```

现在建议调整为：

```text
MCP Server
Agent Security / Guardrails
Memory System
Multi-Agent Patterns
```

原因是 Phase2 已经做过 memory-rag，Phase3 又做了 Agentic RAG。如果下一步继续做记忆，容易重复“检索增强”这条线。

更好的下一步是 MCP。

MCP 可以把前面学到的 Agent 能力接到真实工具系统上：

```text
Agent 不只是回答问题
Agent 能发现可用工具
Agent 能读取资源
Agent 能通过协议调用外部能力
Agent 能在权限边界内工作
```

这会自然引出安全、权限、工具授权和生产化。

---

## 八、Phase4 第一个实战建议

建议做一个当前工程自己的 MCP Server：

```text
ai-agent-learn MCP Server
```

第一版不要做大，先暴露三个工具：

| 工具 | 作用 |
|------|------|
| `search_docs` | 搜索 `docs/` 下的学习文章 |
| `find_code_examples` | 按关键词查找对应阶段的代码脚本 |
| `read_benchmark_summary` | 读取 Phase2 / Phase3 benchmark 结果 |

这不是为了做一个“玩具 MCP”。

它的意义是把当前学习工程变成 Agent 可以操作的知识库：

```text
文档是资源
代码是资源
benchmark 是资源
搜索和读取是工具
权限和边界要显式设计
```

后续可以再加：

```text
读取指定文章
读取指定脚本摘要
列出某个 phase 的学习状态
生成学习复盘
触发 benchmark smoke test
```

但第一版先别急着做太多。MCP 的重点不是工具数量，而是协议、边界和可组合性。

---

## 九、Phase3 Closure Checklist

Phase3 收口前建议确认：

- [x] LangChain / LangGraph 基础文章已整合。
- [x] LangGraph 深入 demo 已完成。
- [x] CrewAI / Claude SDK 已作为对比模块保留。
- [x] framework showdown 已完成。
- [x] Agentic RAG 使用 Phase2 benchmark，不再使用模拟知识库。
- [x] Agentic RAG 输出 JSON、CSV、Markdown 报告。
- [x] `--limit` smoke test 不再覆盖全量 benchmark 产物。
- [x] Phase3 文章已同步最新 benchmark 数字。
- [ ] Phase3 结束后不再新增大规模框架 demo。
- [ ] Phase4 MCP Server 开始前，先写清楚工具接口和权限边界。

---

## 十、下一步怎么走

推荐下一步：

```text
冻结 Phase3 主线
-> 新建 phase-4-advanced/README.md
-> 设计 ai-agent-learn MCP Server
-> 实现最小 MCP 工具集
-> 写一篇 MCP 实战文章
```

这里的“冻结”不是说 Phase3 不能再改，而是不要再扩主线。后续只做必要修正：

```text
修 bug
同步数字
补运行说明
整理文章发布版
```

从学习节奏上看，现在进入 Phase4 是合适的。

Phase3 已经证明：

```text
你不是只会调用框架。
你已经能把 Agent 系统设计成可控、可评估、可复盘的工作流。
```

下一步要证明的是：

```text
这个工作流能不能安全地连接真实工具、真实服务和真实生产环境。
```
