# 混合检索、重排序与 RAG 评估深入解析

> 前置要求：跑过基本 RAG 管道，了解分块、Embedding、检索的基础概念。
> 配套代码：[phase-2-rag/03-hybrid-search/](../../phase-2-rag/03-hybrid-search/) 和 [phase-2-rag/04-rag-evaluation/](../../phase-2-rag/04-rag-evaluation/)

---

混合检索解决了"怎么搜得更全"，重排序解决了"怎么排得更准"，评估解决了"怎么知道系统真的变好了"。这三件事是 RAG 从"能跑"到"能用"的关键跃迁。

整篇文章分两部分：前半部分讲混合检索 + 重排序的技术方案，后半部分讲如何用 RAGAS 等工具量化评估你的 RAG 系统。

---

## 一、为什么需要混合检索？

纯向量检索（Dense Retrieval）虽然擅长语义匹配，但在以下场景表现不佳：

- **精确关键词匹配**：用户搜索 "BM25 算法"，向量检索可能返回语义相关但不包含 "BM25" 的文档
- **专有名词和缩写**：如 "HNSW"、"RRF"、"RAGAS" 等术语，向量模型可能无法准确捕获
- **短查询**：一两个词的查询在向量空间中的表示不够精确

BM25 稀疏检索恰好弥补了这些不足——它基于精确的词频匹配，对关键词敏感。但 BM25 无法理解同义词和语义关系。

**混合检索 = BM25（关键词匹配）+ 向量检索（语义匹配）**，两者互补。

```
查询："Python 装饰器"

BM25 擅长：
  ✓ "Python 装饰器是一种语法糖..."  （精确匹配关键词）
  ✗ "函数包装器模式可以..."          （语义相关但无关键词）

向量检索擅长：
  ✓ "函数包装器模式可以..."          （语义匹配）
  ✗ "Python 装饰器是一种语法糖..."  （可能排名不高）

混合检索：两者都能找到！
```

## 二、BM25 稀疏检索

### 核心公式

BM25 的核心思想：一个词在文档中出现越多（TF），在所有文档中出现越少（IDF），这个词对该文档的重要性就越高。

```
BM25(q, d) = Σ IDF(qi) × (f(qi, d) × (k1 + 1)) / (f(qi, d) + k1 × (1 - b + b × |d|/avgdl))
```

- `f(qi, d)`: 词 qi 在文档 d 中的词频
- `|d|`: 文档长度
- `avgdl`: 平均文档长度
- `k1`: 词频饱和参数（通常 1.2~2.0）
- `b`: 文档长度归一化参数（通常 0.75）

### 中文场景的关键：分词

英文天然以空格分词，但中文需要分词工具。对比：

```python
# 按字符切分（简单但效果差）
list("BM25是信息检索算法")  # ['B', 'M', '2', '5', '是', '信', '息', ...]

# jieba 分词（推荐）
jieba.cut("BM25是信息检索算法")  # ['BM25', '是', '信息', '检索', '算法']
```

jieba 分词能正确识别 "BM25"、"信息检索" 等词组，BM25 的关键词匹配效果显著提升。

## 三、向量稠密检索

### Bi-Encoder 架构

向量检索使用 Bi-Encoder：查询和文档分别编码为向量，通过余弦相似度衡量相关性。

```
Query  → Encoder → q_vec ─┐
                           ├→ cosine_similarity(q_vec, d_vec)
Doc    → Encoder → d_vec ─┘
```

优点：文档向量可以预计算并索引，检索速度快（毫秒级）。
缺点：查询和文档独立编码，无法捕获交互信息，精度有限。

### 常用模型

| 模型 | 维度 | 适用场景 |
|------|------|---------|
| all-MiniLM-L6-v2 | 384 | 英文通用，轻量 |
| BGE-base-zh | 768 | 中文优化 |
| M3E-base | 768 | 中文优化 |
| text-embedding-v3 | 1024+ | 商用 API |

## 四、RRF 排序融合

### 算法原理

Reciprocal Rank Fusion (RRF) 是一种简单有效的排序融合方法：

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

- `k` 是平滑参数（通常取 60），防止排名靠前的文档权重过大
- `rank_i(d)` 是文档 d 在第 i 个检索器中的排名（从 1 开始）

### 为什么用 RRF 而不是分数加权？

BM25 分数和余弦相似度的量纲完全不同，直接加权需要归一化，而归一化方式的选择本身就是个问题。RRF 只用排名，天然避免了这个问题。

```python
def reciprocal_rank_fusion(rankings, k=60):
    fused_scores = {}
    for ranking in rankings:
        for rank, (doc_idx, _score) in enumerate(ranking):
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

## 五、Cross-Encoder 重排序

### 为什么需要重排序？

混合检索的召回率高，但精度有限。Cross-Encoder 通过同时编码查询和文档，捕获细粒度的交互信息，精度远高于 Bi-Encoder。

```
Bi-Encoder:    Query → vec_q    Doc → vec_d    → similarity(vec_q, vec_d)
Cross-Encoder: [Query; Doc] → Encoder → relevance_score
```

### 两阶段架构

```
10000 篇文档
     │
     ▼ 混合检索（BM25 + Dense + RRF）
    ~20 个候选                          ← 速度快，召回率高
     │
     ▼ Cross-Encoder 重排序
    Top 3 最相关                        ← 精度高，速度慢
     │
     ▼ 送入 LLM 生成回答
```

Cross-Encoder 的计算成本是 O(n)（n = 候选数），所以只能用于少量候选的精排，不能直接对全量文档使用。

### 常用模型

| 模型 | 适用场景 |
|------|---------|
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 英文通用，轻量 |
| BAAI/bge-reranker-base | 中文优化 |
| BAAI/bge-reranker-v2-m3 | 多语言 |

## 六、查询改写技术

查询改写解决的是"语义鸿沟"问题——用户的提问方式和文档的表述方式往往不同。

### HyDE（Hypothetical Document Embeddings）

让 LLM 先生成一个假设性答案，用这个答案去检索。因为 LLM 生成的"答案"在表述风格上更接近文档。

```
用户问题: "怎么让检索更准？"
     │
     ▼ LLM 生成假设答案
"提升检索精度可以通过混合检索、重排序、查询改写等技术..."
     │
     ▼ 用假设答案的向量去检索
匹配到更相关的文档
```

### Multi-Query（多查询扩展）

将一个问题改写为多个不同角度的查询，扩大检索覆盖面。

```
原始: "RAG 系统有哪些常见问题？"
     │
     ▼ LLM 改写
1. "RAG 检索增强生成的局限性"
2. "RAG 系统常见的失败模式"
3. "如何诊断 RAG 系统的质量问题"
     │
     ▼ 分别检索，合并去重
覆盖更多相关文档
```

### Step-back Prompting

先问一个更抽象的问题获取背景知识，再回答具体问题。

```
具体问题: "Cross-Encoder 比 Bi-Encoder 慢多少？"
     │
     ▼ Step-back
抽象问题: "Cross-Encoder 和 Bi-Encoder 的架构区别是什么？"
     │
     ▼ 两个问题都检索，合并上下文
获得更全面的背景信息
```

## 七、完整管道架构

把前面的组件串起来，一条生产级 RAG 管道长这样：

```
用户问题
   │
   ▼ 查询改写 (HyDE / Multi-Query / Step-back)
改写后的查询（1~N 个）
   │
   ▼ 混合检索 (BM25 + Dense + RRF)
~20 个候选文档
   │
   ▼ Cross-Encoder 重排序
Top-K 最相关文档
   │
   ▼ 构建上下文 + LLM 生成
带引用的回答
```

每个阶段的作用：

1. **查询改写**：扩大检索覆盖面，弥合语义鸿沟
2. **混合检索**：高召回率的粗筛，兼顾关键词和语义
3. **重排序**：高精度的精排，筛选最相关的文档
4. **生成**：基于高质量上下文生成准确回答

管道搭好了，问题也来了：加入了这么多组件（查询改写、混合检索、重排序），每个组件都有一堆参数可调——怎么知道调参之后系统真的变好了？这就需要系统化的评估。

---

## 八、为什么需要系统化评估？

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

## 九、RAGAS 四大核心指标

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

## 十、评估数据集构建

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
# 简化示例：实际使用时需替换 llm() 为具体的 API 调用
def generate_eval_from_docs(documents: list[str]) -> list[dict]:
    """从文档自动生成评估数据集"""
    for doc in documents:
        question = llm("基于以下文档生成一个问题：" + doc)
        ground_truth = llm("基于以下文档回答问题：" + doc + "\n问题：" + question)
        yield {"question": question, "ground_truth": ground_truth, "source_doc": doc}
```

## 十一、评估驱动的 RAG 优化

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

## 十二、超越 RAGAS：端到端评估

### 延迟评估

用户体验不只看准确性，还看速度：

| 阶段 | 目标延迟 |
|------|---------|
| 检索（BM25 + Dense） | < 100ms |
| 重排序（Cross-Encoder） | < 500ms |
| LLM 生成 | < 3s |
| 端到端 | < 5s |

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

---

## 十三、练习代码

### 混合检索与重排序（03-hybrid-search/）

三个递进式实现：

1. `01_hybrid_retrieval_pipeline.py` — 混合检索基础（BM25 + Dense + RRF），无需 API key
2. `02_rerank_pipeline.py` — 添加 Cross-Encoder 重排序，无需 API key
3. `03_full_rag_pipeline.py` — 完整管道（查询改写 + 混合检索 + 重排序 + LLM 生成），需要 LLM API key

```bash
cd phase-2-rag/03-hybrid-search
pip install -r requirements.txt
python 01_hybrid_retrieval_pipeline.py
python 02_rerank_pipeline.py
cp .env.example .env  # 填入 API key
python 03_full_rag_pipeline.py
```

### RAG 评估（04-rag-evaluation/）

三个递进式实现：

1. `01_ragas_metrics_from_scratch.py` — 手动实现 RAGAS 四大指标，理解底层原理（需要 LLM API）
2. `02_evaluation_pipeline.py` — 自动化评估管道，对比不同 RAG 配置的效果（无需 API，纯本地）
3. `03_rag_optimization_lab.py` — RAG 参数调优实验室，评估驱动的优化循环（需要 LLM API）

```bash
cd phase-2-rag/04-rag-evaluation
pip install -r requirements.txt
cp .env.example .env  # 填入 API key（01 和 03 需要）
python 01_ragas_metrics_from_scratch.py
python 02_evaluation_pipeline.py
python 03_rag_optimization_lab.py
```

### 建议的实践顺序

1. 先跑 `03-hybrid-search/` 的三个脚本，理解混合检索 + 重排序的完整管道
2. 再跑 `04-rag-evaluation/` 的脚本，用量化指标评估你刚才搭的管道
3. 回到混合检索脚本，调参（改 top_k、first_stage_k、分块大小），观察评估指标的变化
