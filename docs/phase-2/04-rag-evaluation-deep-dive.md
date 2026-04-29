# RAG 系统评估深入解析

## 为什么需要系统化评估？

"看起来回答得不错"不是评估标准。RAG 系统有多个可调参数（分块大小、检索策略、top_k、重排序阈值等），每次调整都可能在某些查询上改善效果，在另一些查询上退步。没有量化评估，你无法判断系统是在进步还是退步。

```
调参前：
  "RAG 的核心流程是什么？" → 回答正确 ✓
  "BM25 的 k1 参数作用？" → 回答含糊 △

调参后（增大 top_k）：
  "RAG 的核心流程是什么？" → 回答冗长，混入无关信息 ✗
  "BM25 的 k1 参数作用？" → 回答准确 ✓

没有评估 → 你以为系统变好了
有评估 → 你知道整体分数从 0.78 降到了 0.72
```

## RAGAS 四大核心指标

RAGAS（Retrieval Augmented Generation Assessment）定义了四个正交维度来评估 RAG 系统：

```
                    ┌─────────────────────────────────────┐
                    │         RAG 评估维度                  │
                    │                                     │
  生成质量 ─────────┤  Faithfulness    答案是否忠于上下文    │
                    │  Answer Relevancy 答案是否切题        │
                    │                                     │
  检索质量 ─────────┤  Context Precision 检索结果精确度     │
                    │  Context Recall    检索结果召回率     │
                    └─────────────────────────────────────┘
```

### 1. Faithfulness（忠实度）

核心问题：回答中的每个声明是否都能在检索到的上下文中找到依据？

```
计算方法：
1. 从回答中提取所有事实性声明（claims）
2. 逐一验证每个声明是否被上下文支持
3. Faithfulness = 有依据的声明数 / 总声明数

示例：
  上下文："BM25 基于词频和逆文档频率计算相关性"
  回答："BM25 基于 TF-IDF 计算相关性，由 Robertson 在 1994 年提出"
  
  声明 1: "BM25 基于 TF-IDF 计算相关性" → 有依据 ✓
  声明 2: "由 Robertson 在 1994 年提出" → 无依据 ✗（幻觉！）
  
  Faithfulness = 1/2 = 0.5
```

这是 RAG 最重要的指标。如果 Faithfulness 低，说明 LLM 在"编造"信息，RAG 的核心价值（减少幻觉）没有实现。

### 2. Answer Relevancy（答案相关性）

核心问题：回答是否切题？

```
计算方法（反向验证法）：
1. 从回答反向生成 N 个可能的原始问题
2. 计算生成问题与真实问题的语义相似度
3. Answer Relevancy = 平均相似度

示例：
  问题："BM25 的 k1 参数有什么作用？"
  回答："BM25 是一种信息检索算法，广泛用于搜索引擎..."
  
  反向生成问题：
    "什么是 BM25？"
    "BM25 有哪些应用？"
    "信息检索算法有哪些？"
  
  这些问题和原始问题相似度低 → Answer Relevancy 低
  说明回答跑题了（问的是 k1 参数，答的是 BM25 概述）
```

### 3. Context Precision（上下文精确度）

核心问题：检索到的文档中，有多少是真正有用的？

```
计算方法：
1. 对每个检索到的文档，判断是否与问题相关
2. Context Precision = 相关文档数 / 检索文档总数

示例：
  问题："Cross-Encoder 的计算复杂度？"
  检索到 5 个文档：
    [1] Cross-Encoder 同时编码 query 和 doc... → 相关 ✓
    [2] BM25 是稀疏检索算法...              → 不相关 ✗
    [3] Cross-Encoder 的时间复杂度是 O(n)... → 相关 ✓
    [4] RAG 系统需要评估...                 → 不相关 ✗
    [5] Bi-Encoder 和 Cross-Encoder 对比... → 相关 ✓
  
  Context Precision = 3/5 = 0.6
```

精确度低意味着检索结果中噪声多，LLM 需要从大量无关信息中筛选有用内容，容易被干扰。

### 4. Context Recall（上下文召回率）

核心问题：回答问题所需的所有信息是否都被检索到了？

```
计算方法：
1. 从 ground truth 中提取关键信息点
2. 检查每个信息点是否能在检索到的上下文中找到
3. Context Recall = 被覆盖的信息点数 / 总信息点数

示例：
  问题："混合检索的优势？"
  Ground Truth："混合检索结合 BM25 的关键词匹配和向量检索的语义匹配，
                通过 RRF 融合排序，兼顾精确匹配和模糊匹配"
  
  信息点：
    [1] 结合 BM25 关键词匹配 → 检索到 ✓
    [2] 结合向量检索语义匹配 → 检索到 ✓
    [3] 通过 RRF 融合排序    → 未检索到 ✗
    [4] 兼顾精确和模糊匹配   → 检索到 ✓
  
  Context Recall = 3/4 = 0.75
```

## 评估数据集构建

### 数据集结构

```python
eval_sample = {
    "question": "用户问题",
    "contexts": ["检索到的文档1", "检索到的文档2"],  # RAG 系统实际检索到的
    "answer": "RAG 系统生成的回答",                  # RAG 系统实际生成的
    "ground_truth": "标准答案/参考答案",              # 人工标注的
}
```

### 构建策略

| 方法 | 适用场景 | 成本 |
|------|---------|------|
| 人工标注 | 高质量评估集 | 高 |
| LLM 生成 + 人工审核 | 快速扩充 | 中 |
| 从文档自动生成 QA 对 | 冷启动 | 低 |
| 用户真实查询日志 | 线上评估 | 低（但需积累） |

### 自动生成评估集

```python
def generate_eval_from_docs(documents: list[str]) -> list[dict]:
    """从文档自动生成评估数据集"""
    for doc in documents:
        # 让 LLM 基于文档生成问题
        question = llm("基于以下文档生成一个问题：" + doc)
        # 让 LLM 基于文档生成标准答案
        ground_truth = llm("基于以下文档回答问题：" + doc + "\n问题：" + question)
        yield {"question": question, "ground_truth": ground_truth, "source_doc": doc}
```

## 评估驱动的 RAG 优化

### A/B 测试不同配置

```
配置 A: BM25 only, top_k=5
配置 B: Hybrid (BM25+Dense), top_k=5
配置 C: Hybrid + Rerank, first_stage_k=10, final_k=3

在同一评估集上运行三个配置：

| 指标              | 配置 A | 配置 B | 配置 C |
|-------------------|--------|--------|--------|
| Faithfulness      | 0.72   | 0.78   | 0.85   |
| Answer Relevancy  | 0.80   | 0.82   | 0.84   |
| Context Precision | 0.55   | 0.65   | 0.82   |
| Context Recall    | 0.60   | 0.75   | 0.70   |

结论：配置 C 精确度最高，但召回率略低于 B
→ 可以尝试增大 first_stage_k 来提升召回率
```

### 诊断与调优指南

| 症状 | 诊断 | 处方 |
|------|------|------|
| Faithfulness 低 | LLM 在编造信息 | 强化 Prompt（"只基于上下文回答"）、减少 temperature |
| Answer Relevancy 低 | 回答跑题 | 改进 Prompt 模板、检查检索结果是否相关 |
| Context Precision 低 | 检索噪声大 | 加强重排序、减小 final_k、改进分块 |
| Context Recall 低 | 遗漏关键信息 | 增大 first_stage_k、使用多查询扩展、改进分块 |
| Precision 高 + Recall 低 | 检索太保守 | 增大候选数、放宽相似度阈值 |
| Precision 低 + Recall 高 | 检索太宽泛 | 加强重排序、缩小 top_k |

### 分块策略对评估的影响

```
分块太小（<100 字）：
  → Context Recall 低（一个块装不下完整答案）
  → 需要检索更多块才能覆盖信息

分块太大（>1000 字）：
  → Context Precision 低（块中包含大量无关信息）
  → LLM 上下文被浪费

最佳实践：200-500 字，带 50-100 字重叠
```

## 超越 RAGAS：端到端评估

### 延迟评估

```
用户体验不只看准确性，还看速度：

| 阶段 | 目标延迟 |
|------|---------|
| 检索（BM25 + Dense） | < 100ms |
| 重排序（Cross-Encoder） | < 500ms |
| LLM 生成 | < 3s |
| 端到端 | < 5s |
```

### 成本评估

```
每次查询的成本 = 检索成本 + 重排序成本 + LLM 成本

LLM 成本 ∝ (系统 Prompt + 检索上下文 + 用户问题) × token 单价

减少 final_k 可以降低 LLM 成本，但可能降低 Context Recall
→ 需要在成本和质量之间找平衡
```

### 鲁棒性评估

测试系统在边缘情况下的表现：
- 知识库中没有答案的问题（应该说"我不知道"）
- 模糊/歧义查询
- 多跳推理问题（需要综合多个文档）
- 对抗性查询（试图让系统输出错误信息）

## 练习代码

本目录包含三个递进式实现：

1. `01_ragas_metrics_from_scratch.py` — 手动实现 RAGAS 四大指标，理解底层原理（需要 LLM API）
2. `02_evaluation_pipeline.py` — 自动化评估管道，对比不同 RAG 配置的效果（无需 API，纯本地）
3. `03_rag_optimization_lab.py` — RAG 参数调优实验室，评估驱动的优化循环（需要 LLM API）

运行方式：

```bash
cd phase-2-rag/04-rag-evaluation
pip install -r requirements.txt
cp .env.example .env  # 填入 API key（01 和 03 需要）
python 01_ragas_metrics_from_scratch.py
python 02_evaluation_pipeline.py
python 03_rag_optimization_lab.py
```
