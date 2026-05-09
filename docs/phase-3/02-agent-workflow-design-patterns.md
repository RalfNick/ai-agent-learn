# Agent 工作流设计模式：路由、重试、修复、拒答与可观测性

> Agent 开发能力的核心，不是会调用某个框架 API，而是能把不确定的 LLM 行为放进可控的工程结构里。
>
> 这一篇不讲某个具体框架的教程，而是从 Phase 3 的 Agentic RAG 实验中提炼可复用的工作流设计模式。

---

## 0. 为什么要谈“工作流设计模式”

很多 Agent 教程会从 ReAct 开始：

```text
Thought -> Action -> Observation -> Thought -> ...
```

这个循环很经典，也很适合理解 Agent 的基本行为。

但一旦进入真实系统，你会发现 ReAct 循环只是最底层的执行形态。真正难的是这些问题：

- 工具结果质量差怎么办？
- 检索上下文不足怎么办？
- 模型回答有幻觉怎么办？
- 是否允许无限重试？
- 什么时候应该拒答？
- 失败路径怎么让人看懂？

这些问题都不是“多写一个 Prompt”能解决的。

它们需要工作流设计。

所谓 Agent 工作流设计，就是回答一句话：

> 当系统遇到不确定性时，下一步应该走哪条路径？

---

## 1. 模式一：显式路由

### 1.1 Prompt 不是路由

很多线性 RAG 会在 Prompt 里写：

```text
如果参考资料不足，请说明无法回答。
```

这句话有用，但不够。

因为它没有改变程序结构。

模型仍然会走：

```text
retrieve -> generate -> return
```

是否拒答，完全交给模型一次生成时的自觉。

显式路由的做法是：

```text
retrieve -> context_grade
    -> generate
    -> query_rewrite
    -> abstain
```

这时“资料不足”不再是文本指令，而是状态字段：

```python
context_score: float
context_reason: str
```

再由路由函数决定下一步：

```python
def route_after_context_grade(state, resources) -> str:
    if state["context_score"] >= resources.min_context_score:
        return "generate"
    if state["retry_count"] < resources.max_retries:
        return "rewrite"
    return "abstain"
```

这就是 Agent 工作流和普通 Prompt Engineering 的区别：

- Prompt Engineering：希望模型遵守
- Workflow Design：让判断改变路径

### 1.2 什么时候需要显式路由

只要某个判断会影响系统行为，就应该考虑显式路由。

常见例子：

| 判断 | 可能路径 |
|------|----------|
| 检索结果是否相关 | 生成 / 重写查询 / 拒答 |
| 工具调用是否成功 | 继续 / 重试 / fallback |
| 用户操作是否高风险 | 执行 / 人工确认 |
| 答案是否忠实 | 返回 / 修复 / 拒答 |
| 成本是否超预算 | 继续 / 降级模型 / 停止 |

如果判断只影响文字表达，可以放 Prompt。

如果判断影响下一步动作，就应该进工作流。

---

## 2. 模式二：有预算的重试

### 2.1 重试不是“再跑一遍”

无效重试：

```text
同一个 query
同一个 top_k
同一个 prompt
同一个模型
再跑一次
```

这种重试通常只是在赌随机性。

有效重试应该至少改变一个变量：

- 改写 query
- 扩大候选数量
- 切换检索策略
- 改变 prompt 约束
- 换更强模型
- 请求人类补充信息

在 Phase 3 的 Agentic RAG 中，重试只做一件事：

```text
context_score 低 -> query_rewrite -> retrieve
```

也就是说，重试不是为了“多试一次”，而是为了“换一个检索角度”。

### 2.2 为什么 max_retries = 1

这次实现里，默认：

```python
max_retries = 1
```

原因不是“懒”，而是工程预算。

每次重试至少增加：

1. 一次 query rewrite LLM 调用
2. 一次 hybrid retrieval
3. 一次 rerank
4. 一次 context grading

如果第一次改写后上下文仍然不足，继续重试的边际收益很可能下降。

Agent 设计里有一个很重要的原则：

> 循环必须有预算，预算必须可观测。

预算可以是：

- 最大步数
- 最大重试次数
- 最大 token
- 最大成本
- 最大延迟
- 最大 wall-clock 时间

没有预算的 Agent，很容易从“自主”变成“失控”。

---

## 3. 模式三：质量检查必须改变控制流

很多系统会做事后评估：

```text
generate -> judge -> log score
```

这当然有价值，但它不是 Agentic。

Agentic 的关键是：

```text
generate -> judge
    -> return
    -> repair
    -> abstain
```

也就是说，judge 的结果必须影响下一步。

Phase 3 的 Faithfulness check 使用阈值：

```python
min_faithfulness = 0.86
```

路由：

```python
if faithfulness >= 0.86:
    return END
if repair_count < 1:
    return "repair"
if faithfulness < 0.45:
    return "abstain"
return END
```

这里有一个细节：不是所有低分都直接拒答。

因为有些回答只是包含轻微未支撑扩展，可以通过 repair 收缩。

所以路径设计成：

```text
低于阈值 -> 先修复
修复后仍很差 -> 拒答
修复后一般 -> 可返回但保留分数
```

这比简单的 pass/fail 更接近真实系统。

---

## 4. 模式四：Repair 不是润色

很多人看到 repair，会以为是：

```text
把回答写得更好
```

但在 RAG 系统里，repair 应该是：

```text
把回答写得更少、更稳、更有依据
```

Phase 3 的 repair prompt 核心是：

```text
删除所有参考资料无法支持的声明。
只保留能从参考资料中找到依据的内容。
如果资料不足，直接说明资料不足。
```

这和普通内容润色的方向相反。

普通润色追求：

- 更完整
- 更顺畅
- 更有表达力

RAG repair 追求：

- 更保守
- 更可证
- 更少幻觉

一个好用的判断标准：

> 如果 repair 后答案变长了，你要警惕；如果 repair 后答案更短但更准确，通常是好事。

---

## 5. 模式五：拒答是能力，不是异常

这次全量 benchmark 中，Agentic RAG 触发了 6 次拒答。

拒答样本之一：

```text
参数扫描在 RAG 优化中有什么价值？
```

Trace：

```text
query_analysis:use_original_query
-> retrieve:rag_optimization_lab,p2_hybrid_article
-> context_grade:0.40
-> query_rewrite:1
-> retrieve:rag_optimization_lab,p2_hybrid_article,p2_rag_overview
-> context_grade:0.40
-> abstain
```

这说明系统做了两次尝试：

1. 原始 query 检索
2. query rewrite 后再次检索

两次上下文评分都只有 0.40，于是拒答。

从产品视角看，拒答可能让用户不爽。

但从可信系统视角看，拒答是必要出口。

特别是这些场景：

- 法律
- 医疗
- 财务
- 企业内部知识库
- 技术支持
- 安全操作

在这些场景里，一个“看起来完整但没有依据”的答案，比“我无法可靠回答”危险得多。

---

## 6. 模式六：Trace 是 Agent 的可观测性底座

每次运行都会记录 route trace：

```text
query_analysis:use_original_query
-> retrieve:p2_rag_overview,rag_loading,pdf_learning_assistant
-> context_grade:0.40
-> query_rewrite:1
-> retrieve:p2_rag_overview,rag_loading,pdf_learning_assistant
-> context_grade:0.40
-> abstain
```

这个 trace 至少回答了五个问题：

1. Agent 是否尝试过改写查询？
2. 检索到了哪些 source？
3. 上下文评分是多少？
4. 是在哪一步拒答的？
5. 是检索失败，还是生成失败？

如果没有 trace，你只能看到最终答案。

看到最终答案是不够的。因为 Agent 系统的质量问题常常不在最后一步，而在中间路径：

- 检索错了
- rerank 排错了
- context grader 过严
- repair 没修好
- abstain 阈值过高

Trace 让这些问题变得可定位。

---

## 7. 模式七：指标要和路径一起看

单看指标，Agentic RAG 的结果是：

| 指标 | 线性 RAG | Agentic RAG |
|------|----------|-------------|
| Precision@3 | 0.578 | 0.556 |
| Recall@3 | 0.453 | 0.442 |
| NDCG@3 | 0.524 | 0.516 |
| Faithfulness | 0.910 | 0.970 |
| 平均延迟 | 4597ms | 7139ms |
| 成本 | $0.0296 | $0.0461 |

如果只看检索指标，Agentic RAG 似乎不划算。

如果只看 Faithfulness，Agentic RAG 很成功。

真正应该看的，是“指标 + 路径”：

- 检索指标略降，说明 query rewrite fallback 不一定改善检索。
- Faithfulness 提升，说明 repair/abstain 对可靠性有效。
- 延迟和成本上升，说明质量控制不是免费的。
- 6 次拒答，说明系统更保守。

这才是能指导设计决策的分析。

---

## 8. 一套可复用的 Agent 工作流模板

很多 Agent 任务都可以抽象成下面的模板：

```text
input
  -> analyze
  -> act
  -> grade_result
      -> continue
      -> retry_with_new_strategy
      -> human_review
      -> abstain/fail
  -> generate_output
  -> verify_output
      -> return
      -> repair
      -> abstain/fail
```

对应到不同领域：

| 领域 | act | grade_result | verify_output |
|------|-----|--------------|---------------|
| RAG | 检索 | 上下文相关性 | Faithfulness |
| 代码 Agent | 修改代码 | 测试/静态检查 | Review |
| 数据分析 Agent | 查询数据 | 结果合理性 | 报告一致性 |
| 客服 Agent | 查知识库/工单 | 信息是否覆盖 | 是否合规 |
| MCP Agent | 调工具 | 工具结果是否可信 | 输出是否泄露敏感信息 |

这就是 Phase 3 真正要练的能力：不是记住 LangGraph API，而是学会把任务设计成可控工作流。

---

## 9. 常见设计错误

### 错误一：把所有判断都塞进 Prompt

Prompt 可以约束表达，但不能替代控制流。

如果判断会影响下一步，就应该进入 State 和路由。

### 错误二：重试没有策略变化

同样条件下再跑一次，大多数时候只是浪费 token。

重试必须改变变量。

### 错误三：没有拒答路径

没有拒答路径的系统，最终会在资料不足时编造。

### 错误四：只看最终答案，不看 trace

Agent 的问题往往发生在中间步骤。没有 trace，就没有可调试性。

### 错误五：只追求指标提升，不看成本

Agentic RAG 把 Faithfulness 从 0.910 提到 0.970，但成本从 $0.0296 增加到 $0.0461。这个 tradeoff 必须摆到台面上。

---

## 10. 从需求到图：一套实际设计流程

如果你拿到一个 Agent 需求，不要先问“用什么框架”。

先问下面五个问题：

1. 这个任务有哪些不可避免的不确定性？
2. 哪些不确定性可以通过工具调用降低？
3. 哪些判断会改变后续路径？
4. 哪些失败可以重试，哪些失败应该终止？
5. 最后如何证明这个 Agent 比线性流程更好？

以 Agentic RAG 为例：

| 设计问题 | 对应答案 |
|----------|----------|
| 不确定性是什么 | 检索是否命中、上下文是否足够、答案是否忠实 |
| 可以用什么工具降低 | hybrid retrieval、reranker、faithfulness judge |
| 哪些判断改变路径 | context grade、faithfulness check |
| 哪些失败可以重试 | 检索不足可以 rewrite 一次，答案不忠实可以 repair 一次 |
| 如何证明有效 | 和 linear hybrid_rerank 做 benchmark 对照 |

把这张表写出来之后，图结构基本就自然出现了：

```text
query_analysis
  -> retrieve
  -> context_grade
     -> generate
     -> query_rewrite
     -> abstain
  -> faithfulness_check
     -> end
     -> repair
     -> abstain
```

这个流程适合很多 Agent 任务，不只适合 RAG。

比如代码修复 Agent：

```text
understand_issue
  -> inspect_code
  -> propose_patch
  -> run_tests
     -> finish
     -> debug_failure
     -> ask_human
```

比如数据分析 Agent：

```text
understand_question
  -> inspect_schema
  -> generate_query
  -> validate_query
  -> run_query
  -> interpret_result
  -> check_answer
```

这些任务的共同点是：中间结果会改变下一步。

只要这个条件成立，你就应该考虑显式工作流。

## 11. State 字段怎么设计

Agent workflow 的 State 不是随便放一个 dict。

State 是系统的记忆、接口和调试日志。

一个好 State 至少包含五类字段：

### 11.1 用户输入字段

```python
question: str
question_id: str | None
```

这类字段通常不应该被后续节点修改。

如果需要改写查询，应该新增字段：

```python
current_query: str
rewritten_queries: list[str]
```

不要覆盖原始问题。否则 trace 里你会看不出系统到底从哪里开始偏移。

### 11.2 中间产物字段

```python
query_analysis: dict
retrieved_docs: list[RetrievedDocument]
context_grade: ContextGrade
draft_answer: str
final_answer: str
```

这些字段是节点之间的交接物。

命名上要避免 `result`、`data`、`output` 这种泛化名字。Agent 系统调试已经够难了，字段名不要再制造噪音。

### 11.3 控制字段

```python
retry_count: int
repair_count: int
max_retries: int
max_repairs: int
route_decision: str
```

控制字段决定图是否会循环。

所有循环都必须有预算字段。没有预算字段的 Agent，很容易在异常样本上失控。

### 11.4 评估字段

```python
precision_at_3: float
recall_at_3: float
mrr: float
ndcg_at_3: float
faithfulness: float
```

这些字段不是每次在线请求都必须计算，但 benchmark 模式必须保留。

没有评估字段，Agent 只能靠主观感受迭代。

### 11.5 可观测字段

```python
trace: list[TraceEvent]
latency_ms: float
cost_usd: float
llm_calls: int
```

可观测字段让你回答三个问题：

1. 系统走了哪条路径？
2. 为什么走这条路径？
3. 这条路径花了多少钱？

在 Agent 系统里，这些问题和最终答案同等重要。

## 12. 设计模式的代码模板

下面给一套可以迁移到其他 Agent 项目的模板。

### 12.1 路由函数模板

```python
from typing import Literal


def route_by_quality(state: AgentState) -> Literal["continue", "retry", "abstain"]:
    if state.quality_score >= state.quality_threshold:
        return "continue"

    if state.retry_count < state.max_retries:
        return "retry"

    return "abstain"
```

这个模板的重点是三分支：

- 通过
- 带预算重试
- 终止或拒答

不要只有通过和重试两个分支。

没有终止分支，系统会把所有失败都伪装成成功。

### 12.2 Retry 模板

```python
def retry_with_new_strategy(state: AgentState) -> AgentState:
    last_failure = state.failures[-1]

    new_query = rewrite_query(
        original_query=state.original_query,
        missing_evidence=last_failure.missing_evidence,
        previous_queries=state.rewritten_queries,
    )

    return state.model_copy(update={
        "current_query": new_query,
        "rewritten_queries": [*state.rewritten_queries, new_query],
        "retry_count": state.retry_count + 1,
    })
```

这里最关键的一点是：重试必须换策略。

如果只是同一个输入再跑一次，那不是重试，是重复消耗预算。

### 12.3 Repair 模板

```python
def repair_answer(state: AgentState) -> AgentState:
    repaired = repair_with_evidence(
        question=state.question,
        context=state.retrieved_docs,
        draft_answer=state.draft_answer,
        unsupported_claims=state.unsupported_claims,
    )

    return state.model_copy(update={
        "draft_answer": repaired,
        "repair_count": state.repair_count + 1,
    })
```

Repair 的输入必须包含 `unsupported_claims`。

否则模型很容易把 repair 理解成润色，最后生成更流畅但更危险的答案。

### 12.4 Abstain 模板

```python
def abstain(state: AgentState) -> AgentState:
    answer = (
        "根据当前知识库资料，无法可靠回答这个问题。"
        f"缺失证据：{'; '.join(state.missing_evidence)}"
    )

    return state.model_copy(update={
        "final_answer": answer,
        "is_abstained": True,
    })
```

拒答也应该有结构。

一个好的拒答不是“我不知道”，而是说明为什么不知道、缺什么证据、下一步可以补什么资料。

## 13. 每个模式应该怎么测试

Agent workflow 的测试不能只测 happy path。

你至少需要准备下面这些测试用例：

| 测试类型 | 构造方式 | 期望行为 |
|----------|----------|----------|
| 上下文充足 | 问题直接命中文档 | 不 rewrite，直接生成 |
| 上下文不足但可补救 | 初次查询召回弱，改写后能命中 | rewrite 一次后生成 |
| 上下文不足且不可补救 | 数据集中没有依据 | abstain |
| 答案轻微不忠实 | 生成答案夹带一个无依据断言 | repair 后通过 |
| 答案严重不忠实 | 大部分断言无依据 | repair 后仍失败，abstain |
| 成本异常 | 多次调用 LLM | trace 中记录调用次数和成本 |

这些测试不是为了追求覆盖率数字，而是为了验证控制流是否符合设计。

一个 Agent 工作流最危险的 bug，不是抛异常。

而是它在应该失败时给出了看似合理的成功答案。

## 14. 指标设计：离线指标和在线指标要分开

在 Phase3 benchmark 里，我们关注：

- Precision@3
- Recall@3
- MRR
- NDCG@3
- Faithfulness
- 延迟
- 成本
- 平均重试次数
- 平均修复次数
- 拒答次数

这些是离线指标。

它们回答的是：在标注数据集上，系统是否比 baseline 更值得用。

但如果进入真实产品，还需要在线指标：

| 在线指标 | 说明 |
----------|------|
| 用户追问率 | 答案是否解决问题 |
| 人工转接率 | Agent 是否经常无法处理 |
| 拒答接受率 | 用户是否认可拒答理由 |
| 答案复制率 | 答案是否可用 |
| 负反馈率 | 是否出现明显错误 |
| 单次会话成本 | Agentic 路由是否过度触发 |

离线指标决定能不能上线。

在线指标决定上线后怎么继续迭代。

这也是 Agent 系统和普通脚本最大的区别之一：它不是写完就结束，而是需要持续观测。

## 15. 工作流设计的五个检查问题

写完一个 Agent graph 后，我建议逐项检查：

### 15.1 每个节点是否只有一个职责？

如果一个节点既分析问题、又调用工具、又决定是否重试，说明拆分不够。

可以接受一个节点内部有多个小函数，但对外契约应该单一。

### 15.2 每条条件边是否有可解释的判断依据？

不要写这种路由：

```python
if llm_says_continue:
    return "continue"
```

更好的方式是：

```python
if state.context_grade.score >= state.context_threshold:
    return "generate"
```

LLM 可以参与判断，但判断结果要落到结构化字段上。

### 15.3 每个循环是否有预算？

任何 rewrite、repair、plan revise、tool retry 都应该有上限。

预算可以是次数，也可以是成本、延迟或 token。

### 15.4 是否有失败出口？

失败出口包括：

- abstain
- ask_human
- fallback_to_baseline
- return_partial_result
- create_ticket

没有失败出口的 Agent，最终会把失败包装成幻觉。

### 15.5 trace 是否足够复盘？

如果一个错误答案出现后，你无法回答“它为什么这样答”，说明 trace 不够。

trace 至少要记录：

- 输入
- 节点路径
- 路由判断
- 工具结果
- LLM 调用摘要
- 成本和延迟
- 最终评估结果

## 16. 这些模式如何迁移到其他 Agent

Agentic RAG 只是一个训练样本。

这些模式可以迁移到很多场景。

### 16.1 代码生成 Agent

| RAG 模式 | 代码 Agent 对应 |
|----------|----------------|
| context grading | 判断代码上下文是否足够 |
| query rewrite | 继续搜索相关文件 |
| faithfulness check | 运行测试和静态检查 |
| repair | 根据测试失败修补代码 |
| abstain | 请求人工确认需求 |

代码 Agent 的 `faithfulness` 不一定由 LLM judge 给出，可以由测试结果给出。

这反而更可靠。

### 16.2 数据分析 Agent

| RAG 模式 | 数据分析 Agent 对应 |
|----------|--------------------|
| retrieve | 查 schema、查样例数据 |
| context grading | 判断字段是否足够回答问题 |
| query rewrite | 改写 SQL 或补充过滤条件 |
| faithfulness check | 校验结果是否支持结论 |
| repair | 修改解释或重新查询 |

数据分析 Agent 最常见的问题是“SQL 跑出来了，但解释过度”。

所以它也需要 faithfulness check。

### 16.3 运维排障 Agent

| RAG 模式 | 运维 Agent 对应 |
|----------|----------------|
| retrieve | 拉日志、指标、告警 |
| context grading | 判断证据是否足够定位问题 |
| query rewrite | 查询更细粒度日志 |
| repair | 修正初步诊断 |
| abstain | 升级到人工值班 |

运维 Agent 的拒答路径尤其重要。

错误的自信诊断，比“证据不足”更危险。

## 17. 公众号文章的“干货感”来自哪里

技术文章的干货感不等于字数。

我认为它来自五个东西：

1. 有真实问题，不是为了介绍框架而介绍框架。
2. 有可运行代码，不只是概念图。
3. 有真实数字，不只说“效果更好”。
4. 有失败分析，不只展示成功样例。
5. 有可迁移方法，不只讲当前 demo。

Phase3 的文章应该围绕这五点写。

读者看完之后，应该能带走一套判断方法：

```text
当我设计一个 Agent 时，
我应该如何拆节点，
如何设计 State，
如何设置路由，
如何限制重试，
如何做评估，
以及如何判断复杂度是否值得。
```

这才是比 API 教程更有价值的部分。

## 18. 最后总结

Agent 工作流设计的核心，不是“让模型多思考几步”。

而是：

- 让判断显式化
- 让失败有路径
- 让重试有预算
- 让修复有约束
- 让拒答成为能力
- 让 trace 解释过程
- 让 benchmark 验证取舍

当你能做到这些，才真正从“调用大模型”进入“设计 Agent 系统”。
