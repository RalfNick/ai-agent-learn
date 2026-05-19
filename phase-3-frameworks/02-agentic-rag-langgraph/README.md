# Phase 3: Agentic RAG with LangGraph

这个子项目是 Phase3 的新主线：把 Phase2 的 `hybrid_rerank` RAG baseline 迁移到 LangGraph，自适应地做上下文评分、查询改写、忠实度检查、回答修复和拒答。

## 为什么不是重新做 RAG

Phase2 已经通过 benchmark 证明默认检索配置应采用 `hybrid_rerank`。Phase3 不重复造数据集，而是复用 Phase2 的真实资料、30 个标注问题和评估指标，重点学习 Agent 工作流设计。

## 运行单问

```bash
python3 run_agentic_rag.py --question "RAG 系统如何评估？"
```

## 运行 benchmark

```bash
python3 benchmark_agentic_rag.py --limit 2
python3 benchmark_agentic_rag.py
```

## 输出

- `outputs/agentic_rag_results.json`
- `outputs/agentic_rag_summary.csv`
- `reports/agentic_rag_experiment_report.md`

