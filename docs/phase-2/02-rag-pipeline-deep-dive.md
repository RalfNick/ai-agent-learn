# 从零构建 RAG 管道：检索增强生成的原理与实践

> 本文是 AI Agent 系统性学习系列的第 2 篇。我们从 RAG 的核心问题出发：LLM 为什么需要外部知识？然后一步步构建完整的 RAG 管道，从最简单的 Naive RAG 到混合检索、重排序、查询改写等高级技术，最后用 RAGAS 框架评估系统质量。
>
> 前置要求：了解 LLM 基本概念，完成 Phase 1 的学习。
>
> 配套代码：[phase-2-rag/](../../phase-2-rag/)

---

## 1. 为什么需要 RAG

### 1.1 LLM 的两大痛点

LLM 很强，但有两个根本性的问题：

**知识过时**：模型的知识停留在训练数据的截止日期。你问它"2024 年最新的 AI 框架"，它可能给你 2023 年的信息。

**幻觉（Hallucination）**：当 LLM 不知道答案时，它不会说"我不知道"，而是自信地编造一个看起来合理的答案。这在需要准确性的场景（医疗、法律、金融）是致命的。

### 1.2 RAG 的核心思想

RAG（Retrieval-Augmented Generation，检索增强生成）的思路很直接：

> **不要让 LLM 凭记忆回答，而是先帮它找到相关资料，再让它基于资料回答。**

就像开卷考试 vs 闭卷考试。RAG 把 LLM 从"闭卷"变成了"开卷"。

### 1.3 RAG 的完整流程

```
离线阶段（建索引）：
  文档 → 加载 → 分块 → Embedding → 存入向量数据库

在线阶段（回答问题）：
  用户问题 → Embedding → 向量检索 → 取回相关文档 → 拼接 Prompt → LLM 生成回答
```

接下来我们逐一拆解每个环节。

---

## 2. 文档加载与预处理

RAG 的第一步是把非结构化数据变成可处理的文本。

### 2.1 常见文档格式

| 格式 | 工具 | 注意事项 |
|------|------|----------|
| 纯文本 | 直接读取 | 注意编码（UTF-8） |
| PDF | pypdf / pdfplumber | 表格和图片需要特殊处理 |
| 网页 | BeautifulSoup / trafilatura | 需要去除导航栏、广告等噪声 |
| Markdown | 按标题结构解析 | 天然适合分块 |

### 2.2 文本清洗

垃圾进 → 垃圾出。文本清洗直接影响后续所有环节的质量：

- 去除多余空行和空格
- 去除控制字符
- 去除无意义的短行（页眉页脚等）

```python
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()
```

### 2.3 Document 数据结构

每个文档块不只是文本，还需要携带元数据（来源、页码、标题等），方便后续溯源：

```python
@dataclass
class Document:
    content: str
    metadata: dict  # {"source": "report.pdf", "page": 3}
```

> 配套代码：`01-basic-rag/01_document_loading.py`

---

## 3. 文本分块策略

### 3.1 为什么需要分块

两个原因：
1. **LLM 上下文有限**：不能把整本书塞进 Prompt
2. **检索精度**：块越小，检索越精确（但太小会丢失上下文）

### 3.2 chunk_size 和 chunk_overlap

- `chunk_size`：每个块的目标大小。太大 → 检索不精确；太小 → 上下文不足
- `chunk_overlap`：相邻块的重叠部分。防止关键信息被截断在两个块的边界

经验值：chunk_size 200-500 字符，overlap 10-20%。

### 3.3 四种分块策略

**固定大小分块**：按字符数硬切。简单粗暴，可能在句子中间截断。

**递归字符分块**：先尝试按段落（`\n\n`）切，切不动再按句子（`\n`、`。`）切，最后按字符切。这是 LangChain `RecursiveCharacterTextSplitter` 的核心算法，也是最常用的通用方案。

**Markdown 结构分块**：按标题层级切分，保留文档结构作为元数据。适合技术文档。

**语义分块**：用 Embedding 模型判断相邻句子的语义相似度，在语义断裂处切分。精度最高但实现最复杂。

```
策略选择建议：
  快速原型 → 固定大小
  通用文档 → 递归字符（推荐）
  结构化文档 → Markdown 结构
  高质量 RAG → 语义分块
```

> 配套代码：`01-basic-rag/02_text_chunking.py`

---

## 4. Embedding 与向量数据库

### 4.1 Embedding 的本质

Embedding 把文本映射到高维向量空间。核心性质：**语义相近的文本，向量也相近**。

```
"什么是人工智能？"  →  [0.12, -0.34, 0.56, ...]
"AI 的定义是什么？"  →  [0.11, -0.33, 0.55, ...]  ← 很接近！
"今天天气怎么样？"  →  [0.78, 0.12, -0.45, ...]  ← 很远
```

### 4.2 常用 Embedding 模型

| 模型 | 维度 | 语言 | 特点 |
|------|------|------|------|
| all-MiniLM-L6-v2 | 384 | 英文为主 | 轻量快速，适合学习 |
| BAAI/bge-small-zh-v1.5 | 512 | 中文 | 中文场景推荐 |
| text-embedding-3-small | 1536 | 多语言 | OpenAI API，效果好 |

### 4.3 ChromaDB 向量数据库

ChromaDB 是最简单的向量数据库，适合开发和学习：

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

# 存入文档（自动 Embedding）
collection.add(
    documents=["RAG 是检索增强生成...", "Agent 是智能系统..."],
    ids=["doc_0", "doc_1"],
)

# 语义搜索
results = collection.query(query_texts=["什么是 RAG？"], n_results=2)
```

ChromaDB 内置了 Embedding 功能（默认用 all-MiniLM-L6-v2），也支持自定义 Embedding 模型和持久化存储。

> 配套代码：`01-basic-rag/03_embedding_vectorstore.py`

---

## 5. Naive RAG：最简实现

把前面的组件串起来，就是一个完整的 Naive RAG：

```python
# 1. 索引
chunks = split_text(document)
collection.add(documents=chunks, ids=[...])

# 2. 检索
results = collection.query(query_texts=[question], n_results=3)

# 3. 生成
prompt = f"基于以下文档回答问题：\n{results}\n\n问题：{question}"
answer = llm(prompt)
```

### 5.1 Naive RAG 的局限

这个最简版本能跑，但有明显问题：

1. **检索质量**：纯向量搜索可能遗漏关键词精确匹配的文档
2. **排序质量**：向量相似度 ≠ 真正的相关性
3. **查询理解**：用户的问题可能模糊，直接检索效果差
4. **信息整合**：简单拼接文档，没有智能筛选

这些问题在 02-advanced-rag 中逐一解决。

> 配套代码：`01-basic-rag/04_naive_rag.py`

---

## 6. 混合检索：BM25 + 向量

### 6.1 两种检索的互补性

| | BM25（稀疏检索） | 向量检索（稠密检索） |
|---|---|---|
| 原理 | 关键词匹配（TF-IDF 变体） | 语义相似度 |
| 擅长 | 精确匹配（专有名词、代码） | 模糊匹配（同义词、改述） |
| 弱点 | 无法理解语义 | 可能忽略关键词 |

混合检索的核心思想：**两者互补，覆盖更多相关文档**。

### 6.2 RRF 排序融合

Reciprocal Rank Fusion（RRF）是最常用的融合算法：

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

- `k` 是平滑参数（通常取 60）
- `rank_i(d)` 是文档 d 在第 i 个检索器中的排名
- 不需要对不同检索器的分数做归一化，直接用排名融合

> 配套代码：`02-advanced-rag/05_hybrid_search.py`

---

## 7. 重排序：Cross-Encoder 精排

### 7.1 两阶段架构

```
10000 篇文档 → Bi-Encoder 粗检索 → 20 个候选 → Cross-Encoder 精排 → Top 3
```

**Bi-Encoder**：查询和文档分别编码为向量，用向量距离衡量相关性。文档向量可以预计算，速度快。

**Cross-Encoder**：查询和文档拼接后一起输入模型，直接输出相关性分数。能捕获查询-文档间的细粒度交互，精度远高于 Bi-Encoder，但每个查询-文档对都要重新计算。

### 7.2 为什么需要两阶段

Cross-Encoder 精度高但太慢，不能对所有文档都跑一遍。所以先用快速的 Bi-Encoder/BM25 筛选出候选集，再用 Cross-Encoder 精排。

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[query, doc] for doc in candidate_docs]
scores = model.predict(pairs)
```

> 配套代码：`02-advanced-rag/06_reranking.py`

---

## 8. 查询改写

### 8.1 语义鸿沟问题

用户的提问方式和文档的表述方式往往不同。比如用户问"怎么让搜索更准"，但文档写的是"检索精度优化方法"。

查询改写就是在检索之前，把用户的问题转化为更适合检索的形式。

### 8.2 三种改写技术

**HyDE（Hypothetical Document Embeddings）**：让 LLM 先生成一个假设性答案，用这个答案去检索。因为 LLM 生成的"答案"在表述风格上更接近文档。

**多查询扩展**：把一个问题改写成多个不同角度的查询，合并检索结果。扩大覆盖面。

**Step-back Prompting**：先问一个更抽象的问题获取背景知识，再回答具体问题。

```
原始问题："Cross-Encoder 比 Bi-Encoder 慢多少？"
Step-back："Cross-Encoder 和 Bi-Encoder 的架构区别是什么？"
```

> 配套代码：`02-advanced-rag/07_query_transformation.py`

---

## 9. RAGAS 评估

### 9.1 为什么需要系统化评估

"看起来回答得不错"不是评估标准。RAG 系统需要量化指标来：
- 对比不同配置的效果（chunk_size、检索策略、模型选择）
- 发现系统的薄弱环节
- 持续监控线上质量

### 9.2 RAGAS 四大指标

| 指标 | 中文 | 衡量什么 | 低分意味着 |
|------|------|----------|-----------|
| Faithfulness | 忠实度 | 回答是否基于检索到的文档 | 存在幻觉 |
| Answer Relevancy | 答案相关性 | 回答是否切题 | 回答跑题 |
| Context Precision | 上下文精确度 | 检索结果中相关文档的比例 | 检索噪声大 |
| Context Recall | 上下文召回率 | 是否检索到了所有需要的信息 | 遗漏关键信息 |

### 9.3 评估驱动优化

```
Faithfulness 低 → 改进 Prompt（强调"只基于文档回答"）
Answer Relevancy 低 → 改进 Prompt 模板
Context Precision 低 → 加入重排序、改进分块
Context Recall 低 → 改进检索策略（混合检索、查询改写）
```

> 配套代码：`02-advanced-rag/08_ragas_evaluation.py`

---

## 10. 总结与下一步

### Phase 2 知识图谱

```
RAG 管道
├── 离线索引
│   ├── 文档加载（PDF、网页、文本）
│   ├── 文本分块（递归字符、Markdown 结构、语义）
│   ├── Embedding（sentence-transformers、OpenAI）
│   └── 向量数据库（ChromaDB）
├── 在线查询
│   ├── 查询改写（HyDE、多查询、Step-back）
│   ├── 混合检索（BM25 + 向量 + RRF 融合）
│   ├── 重排序（Cross-Encoder）
│   └── LLM 生成
└── 评估
    └── RAGAS（Faithfulness、Relevancy、Precision、Recall）
```

### 下一步：Phase 3 — 框架实战

Phase 2 我们从底层理解了 RAG 的每个环节。Phase 3 将学习如何用成熟的框架（LangChain、LangGraph、CrewAI、Claude Agent SDK）来构建更复杂的 Agent 系统，把 RAG 作为 Agent 的一个能力模块集成进去。
