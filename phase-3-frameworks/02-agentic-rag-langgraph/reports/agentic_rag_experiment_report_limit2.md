# Agentic RAG 实验报告：LangGraph 是否真的带来编排价值

## 1. 实验设置

本实验复用 Phase2 benchmark 的真实资料和 30 个标注问题，对比两种系统：

| 系统 | 说明 |
|------|------|
| `linear_hybrid_rerank` | Phase2 最优线性 RAG：Hybrid + Cross-Encoder Rerank |
| `agentic_rag_langgraph` | LangGraph 自适应 RAG：检索评分、条件改写、忠实度检查、回答修复、拒答 |

## 2. 指标结果

| 系统 | P@3 | R@3 | MRR | NDCG@3 | Faithfulness | 平均延迟(ms) | 成本 | LLM调用 | 平均重试 | 平均修复 | 拒答 |
|------|-----|-----|-----|--------|--------------|--------------|------|---------|----------|----------|------|
| `linear_hybrid_rerank` | 0.500 | 0.417 | 1.000 | 0.541 | 1.000 | 3057 | $0.0022 | 4 | 0.00 | 0.00 | 0 |
| `agentic_rag_langgraph` | 0.500 | 0.417 | 1.000 | 0.541 | 1.000 | 4856 | $0.0031 | 6 | 0.00 | 0.00 | 0 |

## 3. 关键观察

- 检索指标主要由 Phase2 的 `hybrid_rerank` 决定，Agentic RAG 不应为了“更 Agent”牺牲默认检索质量。
- Agentic RAG 的价值在生成后质量控制：通过 Faithfulness check、repair 和 abstain 把低可信回答显式暴露出来。
- 代价是延迟和 LLM 调用数上升。是否采用 Agentic RAG，应由问题风险、答案可靠性要求和成本预算决定。

## 4. 自适应 Trace 样例

本次运行没有触发 rewrite、repair 或 abstain。需要降低阈值或补充更难的问题集。

## 5. 阶段结论

Phase3 的目标不是证明 LangGraph 永远更快，而是掌握 Agent 工作流设计：什么时候检索，什么时候重写，什么时候修复，什么时候拒答。这个 benchmark 把这些决策从 prompt 里的隐式愿望，变成了图里的显式控制流。
