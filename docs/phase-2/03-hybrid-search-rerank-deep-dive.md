# 混合检索 + Rerank 深入解析

## 为什么需要混合检索？

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

## BM25 稀疏检索

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

## 向量稠密检索

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

## RRF 排序融合

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

## Cross-Encoder 重排序

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

## 查询改写技术

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

## 完整管道架构

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

## 评估与调优

### RAGAS 四大指标

| 指标 | 含义 | 优化方向 |
|------|------|---------|
| Faithfulness | 回答是否基于检索到的文档 | 改进 Prompt、增加上下文 |
| Answer Relevancy | 回答是否切题 | 改进 Prompt 模板 |
| Context Precision | 检索结果中相关文档的比例 | 改进重排序、分块策略 |
| Context Recall | 是否检索到所有需要的信息 | 改进检索策略、查询改写 |

### 调优建议

- **Context Precision 低** → 加强重排序，减小 final_k
- **Context Recall 低** → 增加 first_stage_k，使用多查询扩展
- **Faithfulness 低** → 改进 Prompt，强调"只基于上下文回答"
- **检索延迟高** → 减小 first_stage_k，使用更轻量的 Cross-Encoder

## 练习代码

本目录包含三个递进式实现：

1. `01_hybrid_retrieval_pipeline.py` — 混合检索基础（BM25 + Dense + RRF），无需 API key
2. `02_rerank_pipeline.py` — 添加 Cross-Encoder 重排序，无需 API key
3. `03_full_rag_pipeline.py` — 完整管道（查询改写 + 混合检索 + 重排序 + LLM 生成），需要配置 LLM API key

运行方式：

```bash
cd phase-2-rag/03-hybrid-search
pip install -r requirements.txt
python 01_hybrid_retrieval_pipeline.py
python 02_rerank_pipeline.py
cp .env.example .env  # 填入 API key
python 03_full_rag_pipeline.py
```
