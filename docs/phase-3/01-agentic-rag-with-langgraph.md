# 从线性 RAG 到 Agentic RAG：用 LangGraph 构建可评估的知识库问答工作流

> 这是 AI Agent 系统性学习 Phase 3 的第一篇。前两阶段我们已经手写过 Agent，也做了 RAG benchmark；这一篇不再写玩具 demo，而是把 Phase 2 的真实 RAG 基线升级成一个可路由、可修复、可拒答、可观测的 LangGraph Agentic RAG。
>
> 配套代码：`phase-3-frameworks/02-agentic-rag-langgraph/`
>
> 关键词：LangGraph、Agentic RAG、Hybrid Search、Rerank、Faithfulness、Graph Trace

---

## 0. 先给结论

这次实验对比了两个系统：

| 系统 | 说明 |
|------|------|
| `linear_hybrid_rerank` | Phase 2 最优线性 RAG：Hybrid + Cross-Encoder Rerank |
| `agentic_rag_langgraph` | LangGraph 自适应 RAG：检索评分、条件改写、忠实度检查、回答修复、拒答 |

全量 30 个问题的结果：

| 系统 | P@3 | R@3 | MRR | NDCG@3 | Faithfulness | 平均延迟 | 成本 | LLM 调用 |
|------|-----|-----|-----|--------|--------------|----------|------|----------|
| `linear_hybrid_rerank` | 0.578 | 0.453 | 0.756 | 0.524 | 0.910 | 4597ms | $0.0296 | 60 |
| `agentic_rag_langgraph` | 0.556 | 0.442 | 0.756 | 0.516 | 0.970 | 7139ms | $0.0461 | 98 |

这个结果非常适合学习，因为它不是“Agentic RAG 全面碾压线性 RAG”的爽文结论。

更真实的结论是：

1. **Agent 编排没有自动提升检索质量**。检索指标主要还是由 chunking、BM25、embedding、rerank 决定。
2. **Agentic RAG 的价值在质量控制**。Faithfulness 从 0.910 提升到 0.970，靠的是 check、repair、abstain 这些控制流。
3. **Agentic RAG 有成本**。平均延迟从 4597ms 增加到 7139ms，LLM 调用从 60 次增加到 98 次。
4. **是否使用 Agentic RAG，不是看它酷不酷，而是看业务是否需要“可靠性优先”。**

如果你只记住一句话：

> RAG 解决“资料从哪里来”，Agentic RAG 解决“资料不够好、回答不够可信时系统该怎么办”。

---

## 1. Phase 2 已经解决了什么

在 Phase 2 里，我们做了一个小而真实的 RAG benchmark：

- 42 个真实资料源：文章、Python 脚本、学习笔记
- 30 个手工标注问题
- 对比 naive vector、hybrid、hybrid + rerank、query transform + rerank
- 指标包含 Precision@3、Recall@3、MRR、NDCG@3、Faithfulness、延迟、成本

Phase 2 的关键结论是：当前默认检索策略应该采用 **Hybrid + Cross-Encoder Rerank**。

原因很简单：

- BM25 擅长精确关键词
- Dense retrieval 擅长语义相似
- RRF 融合两者排名
- Cross-Encoder 对候选文档做精排

这条线性管道可以写成：

```text
question
  -> hybrid_search(BM25 + Dense + RRF)
  -> cross_encoder_rerank
  -> build_context
  -> generate_answer
  -> judge_faithfulness
```

它已经不是 naive RAG 了，也不是“随便向量搜一下”的 demo。

但它依然是线性的。

线性意味着什么？

```text
检索不好，也继续生成
回答有幻觉，也只能事后打分
资料不足，也只能靠 Prompt 祈祷模型别编
```

这就是 Phase 3 要解决的问题。

---

## 2. 为什么线性 RAG 不够

假设用户问：

```text
Prompt 引擎在 Agent 中主要解决什么问题？
```

线性 RAG 会做：

```text
retrieve -> rerank -> generate -> return
```

如果只检索到了一个不完整来源，它仍然会生成答案。即使你在 Prompt 里写：

```text
如果资料不足，请说明无法回答。
```

这仍然只是一个软约束。

软约束的问题是：它不改变系统路径。

模型可能遵守，也可能不遵守；即便它遵守，你也很难系统性记录“为什么这次拒答”。

生产级 RAG 需要的是硬控制流：

```text
上下文分数不足 -> 改写查询
改写后仍不足 -> 拒答
回答忠实度不足 -> 修复
修复后仍不足 -> 拒答
```

这就是 Agentic RAG 的核心。

---

## 3. Agentic RAG 不是“让模型更自主”

很多人一听 Agentic RAG，会下意识理解成：

> 让 LLM 自己决定怎么检索、怎么生成、怎么反思。

这个理解很危险。

真正工程化的 Agentic RAG，不是放任模型，而是**把不确定性放进受控图结构里**。

这次 LangGraph 版本的工作流是：

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

对应的能力：

| 节点 | 作用 |
|------|------|
| `query_analysis` | 初始查询分析，默认保留原始问题 |
| `retrieve` | 使用 Phase 2 最优 `hybrid_rerank` 检索 |
| `context_grade` | 判断上下文是否足以回答 |
| `query_rewrite` | 上下文不足时改写查询并重试 |
| `generate` | 基于上下文生成答案 |
| `faithfulness_check` | 判断回答是否忠于上下文 |
| `repair` | 删除上下文不支持的声明 |
| `abstain` | 资料不足或修复失败时拒答 |

注意一个关键设计：**默认不做 query transform**。

Phase 2 benchmark 已经证明，query transform 并不稳定提升 Recall/NDCG。所以这里把它放在 fallback 路径，而不是默认路径。

这也是学习 Agent 设计时很重要的一点：

> 不要因为某个技巧听起来高级，就把它放进默认链路。默认链路应该简单、稳定、可解释；复杂策略应该出现在明确触发条件之后。

---

## 4. State 设计：Agent 工作流的“合同”

LangGraph 的第一件事不是写节点，而是定义 State。

简化后的状态结构如下：

```python
class AgenticRAGState(TypedDict, total=False):
    question: str
    ground_truth: str
    relevant_source_ids: list[str]

    generated_queries: list[str]
    retrieved: list[dict]
    context: str
    context_score: float
    context_reason: str

    answer: str
    faithfulness: float
    faithfulness_reason: str

    retry_count: int
    repair_count: int
    abstained: bool

    route_trace: list[str]
    retrieval_metrics: dict[str, float]
    timings_ms: dict[str, float]
    llm_usage: dict[str, float]
```

这不是为了类型好看，而是为了让系统每一步都可观察。

一个 Agent 系统里最容易丢失的信息，恰恰是“中间过程”：

- 它为什么重试？
- 它重试了几次？
- 它检索到了哪些来源？
- 它为什么拒答？
- 成本花在哪个节点？
- 最后答案的 Faithfulness 是多少？

如果这些信息没有进入 State，后面就很难进入日志、报告和评估。

所以 Phase 3 的一个关键学习点是：

> State 不是临时变量集合，State 是 Agent 系统的审计日志雏形。

---

## 5. 检索节点：复用 Phase 2 最优策略

检索节点没有重新造轮子，而是复用 Phase 2 benchmark 的 `BenchmarkIndex`：

```python
def retrieve(state: AgenticRAGState, resources: AgenticRAGResources) -> dict:
    queries = state.get("generated_queries") or [state["question"]]
    candidates = resources.index.hybrid_search(
        queries,
        top_k=resources.first_stage_k,
    )
    ranked = resources.index.rerank(
        state["question"],
        candidates,
        top_k=resources.top_k,
    )
    context = build_context(resources.index.chunks, ranked)
    ...
```

这个节点做了几件事：

1. 使用 BM25 + Dense + RRF 做粗召回。
2. 使用 Cross-Encoder 对候选重排序。
3. 构建带来源信息的 context。
4. 记录 `retrieved_source_ids`。
5. 计算 Precision@3、Recall@3、MRR、NDCG@3。

这一步很重要：Agentic RAG 的检索能力必须继承 Phase 2 的 benchmark，而不是重新写一个“看起来更 Agent”的低质量检索器。

否则就会出现一种常见误区：

> 为了展示 Agent 编排，把已经验证过的 RAG 能力退化成模拟知识库，然后得出一个很漂亮但没有工程意义的 demo。

这次重构刻意避免了这个问题。

---

## 6. Context Grading：让“资料够不够”改变路径

上下文评分节点的 prompt 很短：

```text
请评估参考资料是否足以回答问题。
只输出 JSON：{"score": 0.0到1.0, "reason": "一句中文理由"}
```

评分标准：

```text
1.0：资料直接覆盖问题所有关键点
0.7：资料覆盖主要问题，少量细节不足
0.4：资料只有部分相关
0.0：资料基本无关
```

对应路由：

```python
def route_after_context_grade(state, resources) -> str:
    if state["context_score"] >= resources.min_context_score:
        return "generate"
    if state["retry_count"] < resources.max_retries:
        return "rewrite"
    return "abstain"
```

这里的默认阈值是：

```python
min_context_score = 0.62
max_retries = 1
```

为什么只重试一次？

因为重试不是免费的。每次 query rewrite 都会多一次 LLM 调用、一次检索、一次评分。如果第一次改写后仍然低分，继续重写很可能只是浪费成本。

这也是 Agent 设计里必须训练的直觉：

> Agent 的循环必须有预算。没有预算的循环，不是自主，是失控。

---

## 7. Faithfulness Check：生成之后不要立刻返回

线性 RAG 的典型结尾是：

```text
generate -> return
```

Agentic RAG 多了一步：

```text
generate -> faithfulness_check
```

Faithfulness check 复用 Phase 2 的 judge：

```python
score, reason, usage = judge_faithfulness(
    resources.llm,
    eval_item,
    context,
    answer,
)
```

路由逻辑：

```python
def route_after_faithfulness(state, resources) -> str:
    if state["faithfulness"] >= resources.min_faithfulness:
        return "end"
    if state["repair_count"] < resources.max_repairs:
        return "repair"
    if state["faithfulness"] < 0.45:
        return "abstain"
    return "end"
```

默认阈值：

```python
min_faithfulness = 0.86
max_repairs = 1
```

这个逻辑背后有两个取舍：

1. 如果低于 0.86，说明答案可能有未支撑声明，先修复。
2. 如果修复后仍然非常低，直接拒答。

这一步把“回答是否可信”从结果评估变成了控制流。

---

## 8. Repair：不是润色，是删掉不该说的话

repair prompt 的核心要求是：

```text
删除所有参考资料无法支持的声明。
只保留能从参考资料中找到依据的内容。
如果资料不足，直接说明资料不足。
```

这和普通“优化回答”完全不同。

普通优化回答通常会让模型更流畅、更完整、更像人写的。

但 RAG repair 的目标是相反的：

- 宁可短一点
- 宁可保守一点
- 宁可承认资料不足
- 不要为了完整而编造

所以我更愿意把 repair 理解成“收缩回答”，不是“扩写回答”。

---

## 9. Abstain：拒答不是失败

本次全量 benchmark 里，Agentic RAG 拒答了 6 次。

一个例子：

```text
文档加载和清洗为什么是 RAG 的上限因素？
```

Trace：

```text
query_analysis:use_original_query
-> retrieve:p2_rag_overview,rag_loading,pdf_learning_assistant
-> context_grade:0.40
-> query_rewrite:1
-> retrieve:p2_rag_overview,rag_loading,pdf_learning_assistant
-> context_grade:0.40
-> abstain
```

这看起来有点反直觉：明明检索到了 `rag_loading`，为什么还拒答？

原因是 context grader 判断资料只覆盖部分问题，不足以支持完整回答。

这里的拒答是否一定正确？不一定。它可能过于保守。

但这正是 benchmark 的价值：它把问题暴露出来了。

你可以继续优化：

- 降低 `min_context_score`
- 增大 `first_stage_k`
- 改进 chunking
- 改进 context grader prompt
- 对拒答样本做单独分析

没有 Agentic trace 时，你只会看到“答案不好”；有 trace 后，你能看到“它在哪一步变差”。

---

## 10. 结果怎么解释

再看一次结果：

| 系统 | P@3 | R@3 | NDCG@3 | Faithfulness | 延迟 | 成本 |
|------|-----|-----|--------|--------------|------|------|
| 线性 RAG | 0.578 | 0.453 | 0.524 | 0.910 | 4597ms | $0.0296 |
| Agentic RAG | 0.556 | 0.442 | 0.516 | 0.970 | 7139ms | $0.0461 |

检索指标略低，主要因为 Agentic RAG 在低质量上下文时会触发 rewrite，而 rewrite 不一定改善检索结果。

Faithfulness 更高，主要因为：

- 低质量上下文会拒答
- 低 Faithfulness 答案会 repair
- repair 后仍不足会退出

延迟和成本更高，原因也清楚：

- 线性 RAG 每题大约 2 次 LLM 调用：生成 + judge
- Agentic RAG 增加 context grading、rewrite、repair

所以这不是“免费午餐”。

Agentic RAG 的适用场景是：

- 企业知识库
- 法务、财务、技术支持等高准确场景
- 需要审计 trace 的系统
- 允许用成本换可靠性的场景

不适合：

- 闲聊
- 低风险内容生成
- 强延迟敏感场景
- 资料质量本身很差但又要求高召回的场景

---

## 11. 可复现方式

单问运行：

```bash
cd phase-3-frameworks/02-agentic-rag-langgraph
python3 run_agentic_rag.py --question "RAG 系统如何评估？"
```

输出包括：

- 答案
- Graph Trace
- Faithfulness
- Latency
- Cost

全量 benchmark：

```bash
python3 benchmark_agentic_rag.py
```

输出：

```text
outputs/agentic_rag_results.json
outputs/agentic_rag_summary.csv
reports/agentic_rag_experiment_report.md
```

---

## 12. 读者可以照着复现的最小闭环

如果这篇文章只是停留在概念层面，它对学习帮助有限。真正应该留下的是一个可以复现的最小闭环：

```text
准备数据集 -> 跑 Phase2 baseline -> 构建 Agentic workflow -> 跑对照实验 -> 分析 trace -> 调整路由策略
```

对应到工程目录：

```text
phase-2-rag/05-rag-benchmark/
├── benchmark_dataset.py
├── benchmark.py
├── outputs/
└── reports/

phase-3-frameworks/02-agentic-rag-langgraph/
├── agentic_rag_graph.py
├── run_agentic_rag.py
├── benchmark_agentic_rag.py
├── outputs/
└── reports/
```

这套结构刻意没有做成一个大而全的平台。学习阶段更重要的是看清每个组件的责任。

如果你想复现，可以按下面顺序跑：

```bash
cd phase-3-frameworks/02-agentic-rag-langgraph

python3 run_agentic_rag.py --question "RAG 系统如何评估？"
python3 benchmark_agentic_rag.py --limit 2
python3 benchmark_agentic_rag.py
```

第一个命令验证单条问题是否能跑通。

第二个命令验证 benchmark pipeline 是否完整。

第三个命令才是全量实验。

为什么要有 `--limit 2`？

因为 Agentic RAG 的失败往往不是语法错误，而是路径错误：

- 该 rewrite 的时候没有 rewrite
- 不该 repair 的时候 repair 了
- 应该拒答的时候编了一个答案
- 检索结果够用，但 context grader 判断过低
- faithfulness judge 太严格，导致答案被过度修复

这些问题不能靠 `py_compile` 发现，必须靠小样本 trace 先看路径。

## 13. 节点契约表：每个节点到底负责什么

Agentic workflow 最容易写乱的地方，是每个节点都想“顺手多做一点”。比如 query analysis 顺手改写问题，retrieval 顺手判断答案，generation 顺手评估忠实度。

一旦这样写，系统会很快变成一团 prompt。

所以我在实现时给每个节点定义了清晰契约：

| 节点 | 输入 | 输出 | 不应该做什么 |
|------|------|------|--------------|
| `query_analysis` | 原始问题 | 问题类型、关键词、是否需要严格依据 | 不检索、不生成答案 |
| `retrieve` | 问题或改写后问题 | top-k 文档、检索指标、来源 | 不判断最终答案 |
| `context_grade` | 问题、检索上下文 | 上下文是否足够、原因、置信度 | 不生成最终答案 |
| `query_rewrite` | 原始问题、失败原因 | 改写查询、retry 次数 | 不无限重写 |
| `generate` | 问题、上下文 | 草稿答案、引用来源 | 不假装已经验证忠实度 |
| `faithfulness_check` | 答案、上下文 | 忠实度分数、问题点 | 不直接改答案 |
| `repair` | 答案、问题点、上下文 | 修复后答案 | 不新增上下文没有的信息 |
| `abstain` | 失败原因 | 拒答说明 | 不输出猜测性答案 |

这张表比代码本身更重要。

因为你真正训练的是“Agent 系统拆分能力”：一个复杂任务，到底应该拆成哪些节点，每个节点如何交接，什么信息能往下传，什么信息不能往下传。

## 14. Graph 路由不是 if else 堆砌

LangGraph 里最有价值的不是 `add_node`，而是条件边。

一个简化后的路由逻辑可以这样理解：

```python
def route_after_context_grade(state: AgenticRAGState) -> str:
    if state.context_grade.is_sufficient:
        return "generate"

    if state.retry_count < state.max_retries:
        return "query_rewrite"

    return "abstain"
```

这段代码看起来普通，但它表达了三个设计判断：

1. 上下文够用，才进入生成。
2. 上下文不够用，但还有预算，才改写查询。
3. 上下文仍然不够用，就拒答。

这不是为了让 Agent 显得“聪明”，而是为了让系统行为可预测。

生成后的路由也是同理：

```python
def route_after_faithfulness(state: AgenticRAGState) -> str:
    if state.faithfulness_score >= state.faithfulness_threshold:
        return "end"

    if state.repair_count < state.max_repairs:
        return "repair"

    return "abstain"
```

这个设计把“答案是否可信”从 prompt 里的软要求，变成了控制流里的硬约束。

线性 RAG 常见写法是：

```text
检索 -> 生成 -> 返回
```

Agentic RAG 的写法是：

```text
检索 -> 判断上下文是否足够 -> 生成 -> 判断答案是否忠实 -> 必要时修复或拒答
```

多出来的不是步骤，而是责任边界。

## 15. Prompt 设计：不要让一个 Prompt 同时承担多个职责

在 Agentic RAG 里，prompt 最容易膨胀。一个常见错误是写出这样的 prompt：

```text
请分析用户问题，判断是否需要检索，改写查询，基于资料回答，并检查答案是否可靠。
```

这个 prompt 看似完整，实际不可控。

因为模型可能同时做了五件事，你无法判断哪一步出错。

更好的方式是把 prompt 拆成多个窄任务。

### 15.1 Query Analysis Prompt

它只判断问题形态：

```text
你需要分析用户问题，输出 JSON：

{
  "query_type": "definition | comparison | procedure | evaluation | troubleshooting",
  "requires_strict_grounding": true,
  "key_terms": ["..."],
  "expected_evidence": "需要哪些类型的资料才能回答"
}

不要回答问题。
不要改写问题。
```

这个 prompt 的价值是给后续节点提供结构化信号。

比如 evaluation 类问题通常需要指标、实验结果、评价方法；troubleshooting 类问题通常需要失败原因和修复步骤。

### 15.2 Context Grading Prompt

它只判断资料是否足够：

```text
给定用户问题和检索到的资料，判断这些资料是否足以回答问题。

输出 JSON：

{
  "is_sufficient": true,
  "score": 0.0,
  "missing_evidence": ["..."],
  "reason": "..."
}

评分标准：
- 0.8 到 1.0：资料直接覆盖问题核心
- 0.5 到 0.8：资料部分相关，但缺关键证据
- 0.0 到 0.5：资料基本不相关

不要生成最终答案。
```

这里的关键不是让 LLM 判断“好不好”，而是要求它说清楚缺什么证据。

缺失证据会进入 query rewrite。

### 15.3 Faithfulness Prompt

它只检查答案是否被上下文支持：

```text
你是 RAG 答案审查器。

给定：
1. 用户问题
2. 检索上下文
3. 候选答案

请判断候选答案中的每个关键断言是否能被上下文支持。

输出 JSON：

{
  "faithfulness": 0.0,
  "unsupported_claims": ["..."],
  "verdict": "pass | repair | abstain"
}

不要补充新知识。
不要重写答案。
```

这个 prompt 的设计重点是 `unsupported_claims`。没有它，repair 节点就不知道应该删什么。

### 15.4 Repair Prompt

Repair prompt 的边界更窄：

```text
请根据 unsupported_claims 修复候选答案。

要求：
- 删除无法被上下文支持的断言
- 保留能被上下文支持的内容
- 不新增上下文之外的信息
- 如果删除后无法回答问题，输出拒答
```

修复不是让答案更漂亮，而是让答案更诚实。

## 16. Trace 怎么看：不要只看最终答案

Agentic RAG 输出 trace 的目的，是让你知道系统为什么走到这个答案。

一个理想 trace 应该至少包含：

```json
{
  "question_id": "q17",
  "mode": "agentic_rag_langgraph",
  "route": [
    "query_analysis",
    "retrieve",
    "context_grade",
    "generate",
    "faithfulness_check",
    "repair",
    "faithfulness_check",
    "end"
  ],
  "retry_count": 0,
  "repair_count": 1,
  "retrieved_sources": [
    "phase-2-rag/05-rag-benchmark/...",
    "docs/phase-2/..."
  ],
  "faithfulness": 0.93,
  "latency_ms": 8120,
  "cost_usd": 0.0017
}
```

看 trace 时，我会按这个顺序问问题：

1. 检索来源是否命中了标注文档？
2. context grader 为什么认为资料足够或不足？
3. 如果发生 rewrite，改写后的查询有没有更接近缺失证据？
4. 生成答案里哪些断言被判定为 unsupported？
5. repair 是删掉 unsupported claims，还是引入了新信息？
6. 最终 faithfulness 提升是否值得额外延迟？

这比单看答案文本可靠得多。

如果只看最终答案，你只能说“看起来不错”。

如果看 trace，你可以说“它为什么不错，以及哪里还有风险”。

## 17. 三类失败样本复盘

Agentic RAG 的学习价值，很大一部分来自失败样本。

下面是这次实验中最值得关注的三类失败。

### 17.1 检索指标略降

结果里 Agentic RAG 的检索指标没有超过线性 baseline：

| 指标 | 线性 RAG | Agentic RAG |
|------|----------|-------------|
| P@3 | 0.578 | 0.556 |
| R@3 | 0.453 | 0.442 |
| NDCG@3 | 0.524 | 0.516 |

这说明一件事：Agentic workflow 本身不会自动提升检索。

检索质量仍然取决于：

- chunk 策略
- embedding 模型
- BM25 权重
- reranker
- query rewrite 是否真的改变召回空间

所以 Phase3 的重点不是“Agent 替代 RAG 优化”，而是“RAG 不确定时，Agent 如何处理风险”。

### 17.2 Faithfulness 提升来自保守策略

Faithfulness 从 0.910 提升到 0.970，主要来自两个机制：

- 对低置信答案进行 repair
- 对资料不足问题 abstain

这也是为什么 Agentic RAG 的答案有时更短。

它不是更会写，而是更少瞎写。

这对知识库问答很重要。企业知识库、技术文档问答、合规场景里，答案短但可靠，通常比答案丰满但混入幻觉更有价值。

### 17.3 成本上升是真问题

Agentic RAG 的 LLM 调用从 60 次增加到 98 次，成本从 $0.0296 增加到 $0.0461。

这不是可以忽略的细节。

如果把它放到生产环境，假设每天 10 万次问答，成本差异会被放大。

所以 Agentic RAG 不应该无脑用于所有问题。

更合理的策略是：

```text
简单事实问答 -> 线性 hybrid_rerank
复杂推理问题 -> Agentic RAG
检索低置信问题 -> Agentic RAG
高风险业务问题 -> Agentic RAG + abstain + human review
```

也就是说，Agentic RAG 本身也应该被路由。

## 18. 调参方法：从阈值开始，而不是从模型开始

很多人优化 Agent 系统时，第一反应是换更强的模型。

但学习阶段更应该先调三个阈值：

| 参数 | 当前策略 | 调低会怎样 | 调高会怎样 |
|------|----------|------------|------------|
| `context_threshold` | 约 0.62 | 更容易生成，幻觉风险增加 | 更容易 rewrite/abstain，召回成本增加 |
| `faithfulness_threshold` | 约 0.86 | 更少 repair，答案更流畅但风险增加 | 更多 repair/abstain，答案更保守 |
| `max_retries` | 1 | 成本可控，但召回失败难补救 | 可能提升召回，但延迟和成本上涨 |

调参时不要一次改很多项。

我建议的顺序是：

1. 固定检索策略为 `hybrid_rerank`。
2. 固定 `max_retries=1`。
3. 扫描 `context_threshold`，观察 rewrite 数量和 Recall。
4. 扫描 `faithfulness_threshold`，观察 repair 数量和 Faithfulness。
5. 最后再考虑是否增加 `max_retries`。

每次改动都要重新生成 summary：

```bash
python3 benchmark_agentic_rag.py
```

然后比较：

```text
Faithfulness 是否提升？
Recall 是否下降过多？
平均延迟是否还能接受？
成本是否符合预期？
trace 中是否出现过度拒答？
```

这就是从“调 prompt”走向“调系统”的关键。

## 19. 公众号文章可以怎么呈现

如果这篇要发公众号，我建议不要直接把所有代码贴满，而是按下面节奏组织：

1. 开头直接给实验结论表。
2. 用一张图解释线性 RAG 和 Agentic RAG 差异。
3. 讲清楚为什么 query transform 不默认启用。
4. 展示 State 和路由代码，让读者看到 Agent 不是玄学。
5. 展示一个 repair trace，让读者看到 Agent 怎么自我纠错。
6. 展示成本和延迟，让文章保持工程诚实。
7. 最后总结：Agentic RAG 的核心不是更自主，而是更可控。

这比“LangGraph 入门教程”更有传播价值。

因为读者看到的不只是 API，而是一个真实工程决策过程。

## 20. 这篇文章真正想训练什么

学 LangGraph，不是为了会写：

```python
graph.add_node(...)
graph.add_edge(...)
```

这些 API 很快就能学会。

真正要训练的是 Agent 系统设计能力：

1. 怎么把不确定性拆成节点？
2. 哪些判断应该进入路由？
3. 哪些失败可以重试？
4. 哪些失败应该拒答？
5. 怎么证明这个设计值得额外成本？

Phase 2 解决的是“怎么让 RAG 检索更好”。

Phase 3 解决的是“当 RAG 仍然不够好时，Agent 系统如何负责任地行动”。

这才是从 RAG pipeline 走向 Agentic RAG 的关键一步。
