# 做完 Agentic RAG 后再比较框架：LangGraph、CrewAI、Claude SDK 的真实边界

> 框架对比不应该从功能表开始，而应该从一个真实系统开始。做完 Agentic RAG 后，LangGraph、CrewAI、Claude SDK 的边界会变得非常清楚。

---

## 0. 先说观点

很多框架对比文章长这样：

| 功能 | LangGraph | CrewAI | Claude SDK |
|------|-----------|--------|------------|
| 工具调用 | 支持 | 支持 | 支持 |
| 多 Agent | 支持 | 支持 | 支持 |
| RAG | 支持 | 支持 | 支持 |

这种表格几乎没有价值。

因为真正决定选型的，不是“是否支持某个功能”，而是：

> 当需求变复杂时，框架暴露了什么，隐藏了什么，你还有没有逃生通道。

这次 Phase 3 重构后，我们不再凭印象比较框架，而是基于一个真实任务：

```text
构建一个可评估的 Agentic RAG：
- 复用真实 RAG benchmark
- 检索后评分
- 低质量时 query rewrite
- 生成后 Faithfulness check
- 不合格时 repair
- 仍不可靠时 abstain
- 输出 trace、延迟、成本
```

这个任务足够具体，也足够能暴露框架边界。

---

## 1. 这个任务到底需要什么能力

先不谈框架，先看需求本身。

Agentic RAG 需要的不是“多个角色一起聊天”，而是一个强控制流系统：

```text
query_analysis
  -> retrieve
  -> context_grade
      -> generate
      -> query_rewrite -> retrieve
      -> abstain
  -> faithfulness_check
      -> repair
      -> abstain
      -> end
```

它需要：

1. 显式 State：保存问题、查询、检索结果、上下文评分、答案、忠实度、成本。
2. 条件路由：不同分数走不同路径。
3. 循环：query rewrite 后回到 retrieve。
4. 质量检查：Faithfulness 不只是日志，而会触发 repair。
5. 失败出口：资料不足时 abstain。
6. 可观测性：每次运行保留 route trace。
7. Benchmark：能和线性 RAG 对比指标。

这些能力决定了框架选择。

---

## 2. 为什么 LangGraph 适合作为主线

LangGraph 的核心抽象是：

```text
State + Node + Edge
```

这刚好对应 Agentic RAG 的核心问题：

- State：系统现在知道什么？
- Node：当前要做什么？
- Edge：下一步去哪？

在这次实现里，State 包含：

```python
class AgenticRAGState(TypedDict, total=False):
    question: str
    generated_queries: list[str]
    retrieved: list[dict]
    context_score: float
    answer: str
    faithfulness: float
    retry_count: int
    repair_count: int
    abstained: bool
    route_trace: list[str]
    timings_ms: dict[str, float]
    llm_usage: dict[str, float]
```

这说明 LangGraph 没有隐藏关键状态。你必须自己定义它。

这既是学习成本，也是优势。

因为真实 Agent 系统里，状态设计就是系统设计本身。

### 2.1 LangGraph 暴露了设计性决策

LangGraph 不会替你决定：

- 什么情况下重写查询
- Faithfulness 阈值是多少
- repair 几次
- 是否拒答
- trace 记录什么

这些都要你写出来。

这听起来麻烦，但它让系统可控。

比如：

```python
def route_after_context_grade(state, resources) -> str:
    if state["context_score"] >= resources.min_context_score:
        return "generate"
    if state["retry_count"] < resources.max_retries:
        return "rewrite"
    return "abstain"
```

这段代码就是设计决策。

而不是一句 Prompt：

```text
如果资料不足，请尝试重新检索。
```

两者的工程可靠性完全不同。

### 2.2 LangGraph 隐藏了机械性工作

LangGraph 帮你处理：

- 图执行
- 节点调度
- 条件边
- 状态传递
- 后续可接 checkpoint

但它没有隐藏业务判断。

这就是一个好抽象：隐藏机械性工作，暴露设计性决策。

---

## 3. CrewAI 为什么不适合作为这条主线

CrewAI 的强项是组织隐喻：

```text
研究员 -> 分析师 -> 撰写者
```

它适合这种任务：

```text
输入一个产品想法，输出市场分析报告。
```

因为这个任务的关键是“谁做什么”。

例如：

```python
market_researcher = Agent(
    role="市场研究员",
    goal="分析目标市场的规模、增长趋势和用户需求",
)

competitor_analyst = Agent(
    role="竞品分析师",
    goal="全面分析竞争格局，找出差异化机会",
)
```

这很自然。

但 Agentic RAG 的关键不是角色，而是控制流。

它不是：

```text
检索员 -> 评分员 -> 写作者
```

这么简单。

它真正需要表达的是：

```text
如果 context_score < 0.62:
    rewrite query
    retrieve again
elif faithfulness < 0.86:
    repair answer
else:
    return answer
```

CrewAI 可以通过 Task description 让模型“尽量这么做”，但它很难把这些条件变成稳定、可测试、可观测的程序路径。

### 3.1 CrewAI 隐藏了我们需要控制的东西

CrewAI 隐藏了：

- 上下文如何传递
- 中间状态如何组织
- 每个 Task 失败后如何恢复
- 条件跳转如何发生

在内容生产场景里，这些隐藏是优点。

在 Agentic RAG 场景里，这些隐藏就成了问题。

因为我们恰恰要控制这些东西。

### 3.2 CrewAI 仍然值得学

这并不代表 CrewAI 没价值。

它很适合：

- 内容创作团队
- 产品分析团队
- 市场研究团队
- 多角色 brainstorming
- 快速验证多 Agent 协作是否有用

所以重构后的 Phase 3 把 CrewAI 放在 `phase-3-frameworks/01-framework-basics/02-crewai-multi-agent/` 作为 reference，而不是主线。

它是一个重要对比项，但不是企业知识库 Agentic RAG 的主编排框架。

---

## 4. Claude SDK 的边界：强在工具循环，不强在编排

Claude SDK 更接近底层 Agent loop 和工具调用层。

它适合：

- 文件读写
- Shell 工具
- 代码审查
- Guardrails
- Handoff
- MCP 工具体系
- 自主操作环境

你可以手写所有控制流：

```python
if context_score < threshold:
    query = rewrite_query(question)
    docs = retrieve(query)

if faithfulness < threshold:
    answer = repair(answer)
```

这非常灵活。

但问题是：所有编排能力都要自己补。

你要自己设计：

- State
- trace
- retry budget
- checkpoint
- branch
- resume
- report
- benchmark integration

这对学习底层 Agent 循环非常有价值。

但如果目标是构建一个复杂、可维护、可观测的工作流，LangGraph 的抽象更合适。

### 4.1 Claude SDK 更像 LangGraph 的互补层

更合理的组合方式是：

```text
LangGraph：负责工作流编排
Claude SDK / MCP：负责工具生态和安全工具调用
```

比如将来做 Phase 4/Phase 6：

- LangGraph 决定什么时候查知识库、什么时候调用工具、什么时候人工确认
- MCP/Claude SDK 提供文件、数据库、搜索、代码操作等工具能力

这不是互斥关系。

---

## 5. 三个框架的真实选型表

| 维度 | LangGraph | CrewAI | Claude SDK |
|------|-----------|--------|------------|
| 核心隐喻 | 状态图 | 团队组织 | 自主工具循环 |
| 最强能力 | 控制流、状态、路由 | 多角色协作原型 | 工具调用、安全边界 |
| 暴露什么 | State、节点、边 | 角色、任务、流程 | API、工具、消息 |
| 隐藏什么 | 图执行细节 | 上下文传递和协调 | API 细节 |
| 适合 Agentic RAG | 很适合 | 不适合作为主线 | 可手写但成本高 |
| 适合内容团队 | 可以但偏重 | 很适合 | 一般 |
| 适合自主代码操作 | 可编排 | 不适合 | 很适合 |
| 学习价值 | Agent 系统设计 | 多 Agent 组织建模 | Agent loop 和安全工具 |

一句话总结：

```text
需要控制流：LangGraph
需要角色协作：CrewAI
需要工具自主行动：Claude SDK
```

---

## 6. 用“逃生舱”测试框架

判断一个框架好不好，不要只看 happy path。

要问：当需求超出框架预设时，你有没有逃生舱？

### 测试一：检索质量不足，需要重新查询

LangGraph：

```text
context_grade -> query_rewrite -> retrieve
```

直接加条件边。

CrewAI：

只能在 Task description 里写“如果资料不足，请尝试重新查询”。是否执行取决于模型。

Claude SDK：

可以手写 if，但状态和 trace 要自己维护。

### 测试二：回答有幻觉，需要修复

LangGraph：

```text
faithfulness_check -> repair -> faithfulness_check
```

自然表达。

CrewAI：

可以加一个 Reviewer Agent，但很难稳定控制“修复后再检查，不通过再拒答”的路径。

Claude SDK：

可以手写循环，但要自己实现预算和日志。

### 测试三：资料不足，需要拒答并记录原因

LangGraph：

`abstain` 是一个明确节点。

CrewAI：

可以让 Agent 输出拒答，但拒答原因和路径不天然结构化。

Claude SDK：

可以手写，但没有内置图级 trace。

这个测试说明：Agentic RAG 的主线选 LangGraph，不是因为它名气大，而是因为它暴露了我们必须控制的东西。

---

## 7. 为什么框架对比必须基于真实 benchmark

如果没有 benchmark，我们很容易得到这种结论：

```text
LangGraph 更可靠
CrewAI 更简单
Claude SDK 更灵活
```

这些都对，但太空。

有了 benchmark，就可以说得更具体：

- LangGraph Agentic RAG 把 Faithfulness 从 0.907 提升到 0.980。
- 代价是延迟从 3269ms 增加到 5108ms。
- 成本从 $0.0296 增加到 $0.0443。
- LLM 调用从 60 次增加到 94 次。
- 系统触发 6 次拒答。

这才是能指导工程决策的对比。

框架不是越强越好，而是要问：

> 这个框架带来的额外可靠性，是否值得它增加的复杂度、延迟和成本？

对企业知识库问答来说，答案通常是：值得，但要有限制地用。

对低风险内容生成来说，答案可能是：不值得。

---

## 8. 重构后的 Phase 3 学习路线

Phase 3 不再是：

```text
学一点 LangGraph
学一点 CrewAI
学一点 Claude SDK
写一篇框架对比
```

而是：

```text
用 LangGraph 做一条真实 Agentic RAG 主线
用 Phase2 benchmark 做基线
用指标证明 Agent 编排的收益和代价
再回头理解 CrewAI / Claude SDK 的边界
```

这个顺序更符合能力成长：

1. 先有真实问题
2. 再设计工作流
3. 再验证指标
4. 最后比较框架

而不是反过来先背框架功能。

---

## 9. 用同一个任务看三种实现模型

框架对比最容易空泛。

为了避免变成“功能表格”，我们用同一个任务来比较：

```text
当 RAG 检索结果质量不足时，系统应该改写查询并重试；
当答案不忠实时，系统应该修复；
当仍然无法可靠回答时，系统应该拒答。
```

这个任务不复杂，但足够暴露框架边界。

### 9.1 LangGraph 写法

LangGraph 的表达方式是图：

```python
workflow.add_node("retrieve", retrieve)
workflow.add_node("context_grade", context_grade)
workflow.add_node("query_rewrite", query_rewrite)
workflow.add_node("generate", generate)
workflow.add_node("faithfulness_check", faithfulness_check)
workflow.add_node("repair", repair)
workflow.add_node("abstain", abstain)

workflow.add_conditional_edges(
    "context_grade",
    route_after_context_grade,
    {
        "generate": "generate",
        "query_rewrite": "query_rewrite",
        "abstain": "abstain",
    },
)
```

这个写法的好处是：失败路径和成功路径一样是系统的一等公民。

你不需要把所有控制逻辑藏在 prompt 里。

### 9.2 CrewAI 写法

CrewAI 更自然的写法是角色协作：

```python
retriever = Agent(role="Retriever", goal="Find relevant evidence")
critic = Agent(role="Critic", goal="Check whether evidence supports the answer")
writer = Agent(role="Writer", goal="Write the final answer")

crew = Crew(
    agents=[retriever, critic, writer],
    tasks=[retrieve_task, critique_task, write_task],
)
```

这种写法非常适合快速演示：

- 角色清晰
- 代码短
- demo 效果好
- 容易向非技术同学解释

但问题在于：精确控制路径会变难。

比如：

- critic 认为资料不足时，retriever 是否必须换 query？
- 最多能重试几次？
- 哪一次重试消耗了多少 token？
- writer 生成后，critic 的判定是否一定触发 repair？
- repair 后是否必须再审查？

这些当然可以做，但会逐渐绕回“自己实现一个状态机”。

这就是我不把 CrewAI 作为 Phase3 主线的原因。

不是它不好，而是它不适合训练这次最核心的能力：可控工作流设计。

### 9.3 Claude SDK 写法

Claude SDK 更像一个工具循环：

```text
model decides -> call tool -> observe result -> decide next action
```

它适合处理开放任务，比如：

- 读代码
- 改代码
- 运行测试
- 根据错误继续修
- 必要时请求人工确认

但在 Agentic RAG benchmark 这种场景里，我们关心的是：

- 是否严格使用 `hybrid_rerank`
- 是否只在低质量检索后 rewrite
- 是否每次 repair 后重新 faithfulness check
- 是否输出统一 trace
- 是否和 baseline 做指标对照

这些要求更接近显式编排，而不是自由工具循环。

所以 Claude SDK 更适合作为高自主执行层，而不是这条主线的编排层。

## 10. 框架选型的评分方式

如果要更系统地比较框架，可以给每个维度设置权重。

对 Agentic RAG 这个任务，我会这样打权重：

| 维度 | 权重 | 原因 |
|------|------|------|
| 控制流显式性 | 25% | 需要清晰表达 rewrite/repair/abstain |
| 状态可观测性 | 20% | benchmark 和 trace 依赖完整状态 |
| 失败恢复能力 | 20% | Agentic RAG 的价值就在失败恢复 |
| 工具集成成本 | 10% | 需要复用 Phase2 检索和 judge |
| 开发速度 | 10% | 学习项目不能陷入框架工程 |
| 可解释性 | 10% | 文章和学习复盘需要解释路径 |
| 生态成熟度 | 5% | 重要，但不是本实验第一目标 |

按这个权重，LangGraph 的优势非常明显。

但如果任务换成“让多个角色协作写一份市场分析报告”，权重就会变化：

| 维度 | 权重变化 |
|------|----------|
| 角色建模 | 提高 |
| 快速原型 | 提高 |
| 精确状态机 | 降低 |
| 离线 benchmark | 降低 |

这时 CrewAI 可能更合适。

如果任务换成“让 Agent 在代码库里自主修复 bug”，权重又会变化：

| 维度 | 权重变化 |
|------|----------|
| 工具循环 | 提高 |
| 文件系统操作 | 提高 |
| 长程自主执行 | 提高 |
| 固定 graph | 降低 |

这时 Claude SDK 或 Codex 类 Agent SDK 更值得研究。

框架没有绝对优劣，只有任务结构和框架表达能力是否匹配。

## 11. 判断框架是否合适的五个逃生问题

我现在会用五个问题判断一个 Agent 框架是否适合当前任务。

### 11.1 我能不能强制某个检查节点必须执行？

在 Agentic RAG 中，faithfulness check 不能依赖模型“自觉”。

它必须每次生成后执行。

如果框架让这个约束很难表达，就不适合高可靠问答。

### 11.2 我能不能限制循环次数？

所有 rewrite、repair、retry 都必须有预算。

如果框架默认鼓励开放循环，而预算控制需要额外绕路，那它更适合探索任务，不适合可控任务。

### 11.3 我能不能拿到完整状态？

benchmark 需要知道：

- 检索到了哪些文档
- 每个节点花了多久
- 调用了几次 LLM
- 哪条路径被触发
- 为什么拒答

如果框架只返回最终答案，就不适合做严肃评估。

### 11.4 我能不能复用已有工程组件？

Phase2 已经有 hybrid retrieval、reranker、faithfulness judge。

Phase3 不应该重新造这些轮子。

框架如果要求把所有东西包装成自己的抽象，迁移成本会很高。

### 11.5 我能不能解释失败？

一个 Agent 系统上线后，最常被问的不是“它怎么成功的”，而是“它为什么错了”。

如果框架不能帮助解释失败路径，那它只适合 demo，不适合长期维护。

## 12. 学习路径应该怎么安排

重构后的 Phase3 不应该平均用力学习三个框架。

更好的学习路径是：

### 12.1 第一阶段：LangGraph 主线

目标不是学会所有 API，而是完成一个真实可评估工作流。

学习重点：

- State schema
- Node contract
- Conditional edge
- Retry budget
- Checkpoint
- Trace
- Benchmark

输出物：

- 一个可运行 Agentic RAG
- 一份 benchmark report
- 一篇系统设计文章

### 12.2 第二阶段：CrewAI 对比

目标不是复制 Agentic RAG，而是理解角色协作范式。

可以做一个较小练习：

```text
让 Researcher、Analyst、Writer 三个角色基于同一批资料输出技术调研摘要。
```

观察重点：

- 角色分工是否自然
- task handoff 是否清晰
- 中间结果是否容易观察
- 失败恢复是否容易控制

输出物：

- 一篇“CrewAI 适合什么、不适合什么”的文章

### 12.3 第三阶段：Claude SDK 对比

目标是理解工具循环和自主执行边界。

可以做一个练习：

```text
让 Agent 阅读 benchmark 失败样本，提出代码修改建议，并运行 smoke test。
```

观察重点：

- 工具调用是否自然
- 文件操作边界是否清楚
- 人工确认点如何设计
- 如何防止自主操作越界

输出物：

- 一篇“工具循环型 Agent 和工作流型 Agent 的边界”的文章

## 13. 从学习项目走向真实项目，还缺什么

Phase3 的 Agentic RAG 已经能证明工作流编排价值，但距离生产系统还有距离。

缺口主要有五个。

### 13.1 更强的可观测性

当前 trace 已经记录路径、重试、修复、成本和延迟。

生产里还需要：

- request id
- user id hash
- session id
- prompt version
- retrieval config version
- model version
- judge version
- error taxonomy

否则线上问题很难归因。

### 13.2 更稳定的评估集

30 个问题足够学习，但不够做生产回归。

真实项目至少需要：

- 高频问题集
- 难例问题集
- 安全边界问题集
- 新文档增量问题集
- 历史失败回归集

这会在 Phase5 或 Phase6 更重要。

### 13.3 Human-in-the-loop

当前系统只有自动 repair 和 abstain。

生产里还需要人工介入路径：

```text
低风险问题 -> 自动回答
中风险问题 -> 自动回答 + 记录
高风险问题 -> 拒答或人工确认
资料冲突问题 -> 请求人工裁决
```

LangGraph 的 checkpoint 和 interrupt 能支持这个方向。

### 13.4 权限和数据边界

企业知识库 Agent 不能只考虑答案质量。

还要考虑：

- 用户是否有权访问某个文档
- 检索阶段是否过滤权限
- trace 中是否泄露敏感内容
- judge prompt 是否包含不该外发的数据
- 日志保留周期如何设置

这部分会自然进入 Phase4/Phase5。

### 13.5 多 Agent 是否真的必要

很多任务不需要多 Agent。

如果一个显式 workflow 已经能解决问题，就不要急着引入多个角色。

多 Agent 适合：

- 不同角色有真实不同工具
- 不同角色有真实冲突目标
- 中间产物需要审查和协作
- 任务可以并行

如果只是把一个流程拆成三个名字不同的角色，通常只是增加复杂度。

## 14. 框架比较文章的写法建议

如果这篇要发公众号，我建议标题不要写成：

```text
LangGraph、CrewAI、Claude SDK 全面对比
```

这种标题容易把文章带向 checklist。

更好的标题是：

```text
做完一个 Agentic RAG 后，我重新理解了 LangGraph、CrewAI 和 Claude SDK
```

这个标题更准确。

文章结构可以是：

1. 先给真实任务和 benchmark 数字。
2. 说明为什么这个任务需要显式控制流。
3. 展示 LangGraph 如何表达 rewrite/repair/abstain。
4. 解释 CrewAI 为什么适合角色协作，但不适合这条主线。
5. 解释 Claude SDK 为什么适合工具循环，但不是主要编排层。
6. 给出框架选型表。
7. 给出后续学习路线。

这篇文章的价值不是“我知道三个框架”，而是“我知道什么时候不该用某个框架”。

这对读者更有帮助。

## 15. 最后总结

如果你问我：学 Agent 框架到底应该学什么？

我的答案不是：

```text
学会 LangGraph、CrewAI、Claude SDK 的 API。
```

而是：

```text
学会判断一个任务需要什么状态、哪些节点、哪些路由、哪些失败恢复路径，以及如何用 benchmark 证明这个设计是否值得。
```

这就是具体 Agent 设计和开发能力。

框架只是表达这些设计的工具。

真正重要的是：你能不能看见系统里的不确定性，并把它变成可控的工程路径。
