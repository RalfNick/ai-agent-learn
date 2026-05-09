# RAG 优化实验报告：从能跑到可证明变好

> Phase 2 验收 benchmark。实验资料来自本仓库真实学习文章和技术脚本，问题集为人工构造并标注相关资料来源。

## 1. 实验设置

- 资料源数量：42
- 文档块数量：527
- 评估问题数量：30
- Embedding 模型：`paraphrase-multilingual-MiniLM-L12-v2`
- Rerank 模型：`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Rerank 实际模式：`cross_encoder`
- LLM 模型：`deepseek/deepseek-chat`

## 2. 对比配置

| 配置 | 说明 |
|------|------|
| `naive_vector` | 只使用向量检索 Top-3 |
| `hybrid` | BM25 + Dense 检索，使用 RRF 融合 |
| `hybrid_rerank` | Hybrid 粗召回后使用 Cross-Encoder 精排 |
| `query_transform_rerank` | LLM Multi-Query 改写 + Hybrid + Cross-Encoder 精排 |

## 3. 指标结果

| 配置 | Precision@3 | Recall@3 | MRR | NDCG@3 | Faithfulness | 平均延迟(ms) | 估算成本 | LLM调用 |
|------|-------------|----------|-----|--------|--------------|--------------|----------|---------|
| naive_vector | 0.550 | 0.446 | 0.728 | 0.495 | 0.863 | 3770 | $0.0287 | 60 |
| hybrid | 0.550 | 0.448 | 0.772 | 0.513 | 0.810 | 3714 | $0.0298 | 60 |
| hybrid_rerank | 0.578 | 0.453 | 0.756 | 0.524 | 0.917 | 3945 | $0.0300 | 60 |
| query_transform_rerank | 0.539 | 0.411 | 0.728 | 0.485 | 0.930 | 4934 | $0.0317 | 90 |

## 4. 结论

- Recall@3 最好：`hybrid_rerank`，分数 `0.453`。
- NDCG@3 最好：`hybrid_rerank`，分数 `0.524`。
- Faithfulness 最好：`query_transform_rerank`，分数 `0.930`。

这组数据说明：Phase 2 的优化不是“感觉更高级”，而是可以量化证明。

相对 `naive_vector`，`hybrid_rerank` 的 Precision@3 从 `0.550` 提升到 `0.578`，Recall@3 从 `0.446` 提升到 `0.453`，NDCG@3 从 `0.495` 提升到 `0.524`，Faithfulness 从 `0.863` 提升到 `0.917`。这说明混合检索加重排序既改善了排序质量，也让生成答案更忠于上下文。

`query_transform_rerank` 的 Faithfulness 最高，达到 `0.930`，但 Recall@3 为 `0.411`，NDCG@3 也低于 `hybrid_rerank`。在这个数据集里，LLM 查询改写没有带来稳定的检索收益，反而可能把部分查询改写偏。它适合作为短查询、口语化查询、术语不稳定场景下的按需增强，而不适合作为默认必开选项。

因此，Phase 2 当前推荐默认配置是：**Hybrid + Cross-Encoder Rerank**。它在质量、延迟、成本之间最均衡。

## 5. 成本与延迟观察

`query_transform_rerank` 会额外调用 LLM 做查询改写，因此 LLM 调用数从 60 次增加到 90 次，平均延迟从 `hybrid_rerank` 的 `3945ms` 增加到 `4934ms`，估算成本从 `$0.0300` 增加到 `$0.0317`。

本实验的成本估算优先使用 LiteLLM 的返回结果；当模型价格表返回 0 时，脚本使用内置 fallback 费率估算。可通过 `BENCHMARK_INPUT_COST_PER_1M` 和 `BENCHMARK_OUTPUT_COST_PER_1M` 覆盖。

结论很清楚：如果目标是默认生产配置，先选择 `hybrid_rerank`；如果遇到召回不足的复杂自然语言问题，再针对性开启 query transform，并用同一套 benchmark 验证它是否真的提升。

## 6. 复现实验

```bash
cd phase-2-rag/05-rag-benchmark
python3 benchmark.py
```

完整明细见：

- `outputs/benchmark_results.json`
- `outputs/benchmark_summary.csv`
