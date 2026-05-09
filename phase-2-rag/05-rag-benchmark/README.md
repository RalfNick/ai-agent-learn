# Phase 2 RAG Benchmark

这个目录是进入 Phase 3 前的 Phase 2 验收实验：用真实项目资料、人工标注问题集和 LLM-as-Judge，把 RAG 从“能跑”升级为“能证明变好”。

## 数据集

- 资料：`benchmark_dataset.py` 中登记了 42 个真实资料源，来自 `docs/` 文章和 `phase-1/phase-2` 技术脚本。
- 问题：30 个手工构造问题，每个问题标注 `relevant_source_ids`、`ground_truth` 和证据说明。

## 对比配置

1. `naive_vector`: 向量检索 Top-K
2. `hybrid`: BM25 + Dense + RRF
3. `hybrid_rerank`: Hybrid 粗召回 + Cross-Encoder 重排序
4. `query_transform_rerank`: LLM Multi-Query 改写 + Hybrid + Cross-Encoder 重排序

## 指标

- Retrieval: `Precision@3`, `Recall@3`, `MRR`, `NDCG@3`
- Generation: `Faithfulness`，由 LLM-as-Judge 对答案和检索上下文打分
- 工程指标：平均延迟、LLM 调用成本估算

## 运行

```bash
cd phase-2-rag/05-rag-benchmark
pip install -r requirements.txt
python3 benchmark.py
```

脚本会读取本目录、`../04-rag-evaluation`、`../03-hybrid-search`、`../01-basic-rag` 里的 `.env`，也会读取仓库根目录 `.env`。默认模型为 `LLM_MODEL=deepseek/deepseek-chat`。

## 输出

- `outputs/benchmark_results.json`: 完整实验明细
- `outputs/benchmark_summary.csv`: 配置级指标表
- `reports/rag_optimization_experiment_report.md`: 可直接作为 Phase 2 验收报告的 Markdown 文章
