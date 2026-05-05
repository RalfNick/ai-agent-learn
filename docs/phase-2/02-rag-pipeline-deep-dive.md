# RAG 管道实战：从跑通到跑好

> 前置要求：跑完 Phase 1，用过 LLM API。
> 配套代码：[phase-2-rag/](../../phase-2-rag/)

---

## 1. LLM 的两个硬伤，RAG 怎么救

LLM 有两个问题，不是"不够强"，而是"根本性的"：

**第一，知识有截止日期。** 训练数据停在某个时间点，之后的它一概不知。你问它最新信息，它要么回答不了，要么硬编。

**第二，它会编。** 专业术语叫幻觉（Hallucination），说白了就是——不知道的时候不会说"不知道"，而是自信地编一个听起来像那么回事的答案。这在医疗、法律、金融场景里是致命的。

RAG 的思路简单到一句话就能说完：**别让模型凭记忆答题，先帮它找到资料，再让它照着资料回答。** 闭卷变开卷。

整条管道长这样：

```
离线（建索引）：  文档 → 加载清洗 → 分块 → Embedding → 向量库
在线（回答）：    问题 → Embedding → 检索 → 拿回相关文档 → 拼 Prompt → LLM 生成
```

下面逐个拆开，重点是**实际写代码时会踩的坑**。

---

## 2. 文档加载：脏数据是万恶之源

### 2.1 格式与工具

| 格式 | 用什么 | 坑 |
|------|--------|-----|
| 纯文本 | `open().read()` | 编码问题，UTF-8 不是万能的 |
| PDF | pypdf / pdfplumber | 表格和图片基本没救，别指望 |
| 网页 | trafilatura（比 BeautifulSoup 省事） | 导航栏、广告、推荐列表全是噪声 |
| Markdown | 按 `#` 层级解析 | 还好，格式相对干净 |

### 2.2 清洗：不做就等着翻车

一句话：**垃圾进，垃圾出。** 下面这段基本够用：

```python
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)          # 多余空行合并
    text = re.sub(r" {2,}", " ", text)               # 多余空格合并
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # 踢掉控制字符
    return text.strip()
```

如果你处理的是网页，还得额外去掉脚本标签、样式标签、导航链接。这块没有通用解法，拿到数据先看几眼再决定清洗策略。

### 2.3 别忘了元数据

每个文档块不光要有文本，还得带着"这东西从哪来的"——后面出问题才能溯源：

```python
@dataclass
class Document:
    content: str
    metadata: dict  # {"source": "report.pdf", "page": 3, "title": "..."}
```

> 配套代码：`01-basic-rag/01_document_loading.py`

---

## 3. 分块：RAG 的第一个关键调参点

### 3.1 为什么要分

两个硬约束：
- LLM 上下文窗口有限，塞不进整本书
- 块太大检索不准，块太小丢上下文

### 3.2 chunk_size 和 chunk_overlap

这俩参数直接影响检索质量，没有银弹，只有经验值：

- **chunk_size**：200-500 字符是比较舒服的范围。英文可以偏大（token 密度低），中文偏小。
- **chunk_overlap**：10%-20%。作用是在块边界处留"缓冲区"——防止一句关键信息正好被切成两半。

### 3.3 四种切法，什么时候用哪个

**固定大小**：按字符数硬切。最粗暴，可能在句子中间一刀两断。只适合快速原型。

**递归字符切分**：先尝试按段落（`\n\n`）切，切不动再按句子（`\n`、`。`），再切不动就按字符。这是 `RecursiveCharacterTextSplitter` 的逻辑，也是大多数场景的默认选择。你不需要自己想切分顺序，它帮你降级。

**按 Markdown 结构切**：识别 `#`、`##` 标题层级，保持文档结构。适合技术文档、wiki 这类有明确层级的内容。

**语义切分**：用 Embedding 模型判断相邻句子是否"在说同一件事"，在语义断裂处下刀。效果最好，但也最慢、最复杂。

```
选哪种？
  原型验证 → 固定大小，先把 pipeline 跑通
  通用场景 → 递归字符，够用了
  结构化文档 → Markdown 结构切分
  线上追求质量 → 语义切分，值得投入
```

> 配套代码：`01-basic-rag/02_text_chunking.py`

---

## 4. Embedding 和向量库：把"语义相似"变成"距离近"

### 4.1 Embedding 干了什么

直说：Embedding 模型把一段文本压成一个固定长度的浮点数数组（向量）。关键性质是——**语义越近的文本，向量之间的距离越近**。

```
"什么是人工智能？"  →  [0.12, -0.34, 0.56, ...]
"AI 的定义是什么？"  →  [0.11, -0.33, 0.55, ...]  ← 向量很接近
"今天天气怎么样？"  →  [0.78, 0.12, -0.45, ...]  ← 向量很远
```

这意味着：**检索 = 在向量空间里找最近邻**。不需要关键词匹配，不需要同义词词典——模型替你做了。

### 4.2 模型怎么选

| 模型 | 维度 | 语言 | 什么时候用 |
|------|------|------|-----------|
| all-MiniLM-L6-v2 | 384 | 英文为主 | 本地开发、快速验证 |
| BAAI/bge-small-zh-v1.5 | 512 | 中文 | 中文场景首选 |
| text-embedding-3-small | 1536 | 多语言 | 调用 API，效果最好但要花钱 |

### 4.3 ChromaDB：够用就好

ChromaDB 是向量数据库里门槛最低的。关键是它自动帮你做 Embedding——你存文本，它自己编码成向量，不用手动调 `model.encode()`：

```python
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()  # 内存模式，进程结束就没
collection = client.create_collection(
    name="my_docs",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},  # 用余弦相似度
)

# 写：直接传文本，自动向量化
collection.add(
    documents=["RAG 是检索增强生成...", "Agent 是智能系统..."],
    ids=["doc_0", "doc_1"],
)

# 读：传查询文本，自动编码 + 搜索
results = collection.query(query_texts=["什么是 RAG？"], n_results=2)
```

生产环境把 `Client()` 换成 `PersistentClient(path="./chroma_db")`，数据就落盘了。

### 4.4 为什么不能暴力扫描

100 万篇文档，384 维向量，暴力算 100 万次余弦相似度——几百毫秒，好像还行？但到 1000 万、1 亿篇时就是秒级延迟了，用户等不了。

ChromaDB 内部用 **HNSW**（分层可导航小世界图）做近似最近邻搜索。原理不用深究，知道效果就行：把 O(N) 降到 O(log N)，千万级也能毫秒出结果。代价是牺牲了微小的精度（ANN 不是精确搜索），但对 RAG 场景来说完全可接受。

> 配套代码：`01-basic-rag/03_embedding_vectorstore.py`

---

## 5. Naive RAG：30 行跑通全流程

前面三个组件串起来就是 Naive RAG：

```python
# 1. 建索引
chunks = split_text(document)
collection.add(documents=chunks, ids=[...])

# 2. 检索
results = collection.query(query_texts=[question], n_results=3)

# 3. 生成
prompt = f"基于以下文档回答问题：\n{results}\n\n问题：{question}"
answer = llm(prompt)
```

这 30 行能跑，但只是"能跑"。实际用起来问题很多：

- 纯向量检索碰到专有名词、代码、缩写时抓瞎（"HNSW" 这种缩写向量模型不一定认识）
- 向量相似度高 ≠ 真正相关。排名靠前的文档可能刚好高频提到某个词，并非真的有用
- 用户问法千奇百怪，直接拿原问题去搜效果不稳定
- 文档简单拼接，没有去重、没有筛选、没有排序

下面几节就是逐一解决这些问题。

> 配套代码：`01-basic-rag/04_naive_rag.py`

---

## 6. 混合检索：BM25 + 向量，两条腿走路

### 6.1 两种检索不是竞争关系，是互补

纯向量检索最怕的场景：用户搜索一个精确术语（比如 "BM25"），向量模型可能返回语义相关但不包含这个词的文档。反过来，纯关键词匹配（BM25）没法理解"跑得快"和"速度快"是一个意思。

两种方法各有盲区，合在一起刚好互补：

| | BM25 | 向量检索 |
|---|------|---------|
| 怎么工作 | 数词频、算逆文档频率 | 语义空间找近邻 |
| 最擅长 | 专有名词、代码、缩写 | 同义词、改写、模糊表达 |
| 搞不定 | "跑得快" 和 "速度快" 它以为是两回事 | "HNSW" 这种缩写它可能不认识 |

### 6.2 RRF：不用归一化的融合

BM25 打分和余弦相似度打分，量纲完全不一样，没法直接加权。最干净的解法是 **RRF（Reciprocal Rank Fusion）**——不看分数看排名：

```
RRF_score(文档) = Σ 1 / (k + 排名)
```

k 一般取 60，作用是防止排名靠前的文档权重过大。本质就是：**在两个检索器里都排前面的文档，大概率真的相关**。

因为只用排名不用分数，彻底避开了归一化的问题。这个方法简单到令人发指，但效果出奇好。

---

## 7. 重排序：让 Cross-Encoder 做精排

### 7.1 Bi-Encoder vs Cross-Encoder

混合检索之后，你拿到了 ~20 个候选文档。但这 20 个里哪些才是真正相关的？这时候该 Cross-Encoder 上场了。

两者的区别：

- **Bi-Encoder**：Query 和 Doc 分别编码成向量，用向量距离衡量相关性。Doc 向量可以**提前算好存起来**，检索时只算 Query 的向量，毫秒级。
- **Cross-Encoder**：把 Query 和 Doc **拼在一起**送进模型，模型直接输出一个相关性分数。因为 Query 和 Doc 在模型内部有充分的交互，精度远高于 Bi-Encoder。代价是——每对 (Query, Doc) 都得从头算一遍。

### 7.2 两阶段架构

所以实际的检索架构是两阶段的：

```
全量文档（1万+）
    │
    ▼ BM25 + 向量 + RRF（快，召回率高）
~20 个候选
    │
    ▼ Cross-Encoder（慢但准）
Top 3-5，送入 LLM
```

Cross-Encoder 太慢不能扫全量，但扫 20 个候选足够了。这个"粗筛 + 精排"的套路在搜索系统里用了十几年，RAG 也不例外。

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[query, doc] for doc in candidate_docs]
scores = model.predict(pairs)  # 每个候选文档拿一个相关性分数
```

---

## 8. 查询改写：用户不会按你的文档风格提问

### 8.1 问题在哪

用户问"怎么让搜索更准"，但你的文档写的是"检索精度优化方法"。词不一样，意思一样——向量模型能 handle 一部分，但不够。

查询改写就是在检索**之前**，把用户的问题改写成更容易命中文档的形式。

### 8.2 HyDE：凭空编一个答案去搜

思路很"反直觉"但很有效：先让 LLM 编一个假设性的答案，用这个答案的向量去检索。因为 LLM 编的"答案"风格上更像知识库里的文档，比用户的口语化问题更容易命中。

```
用户："怎么让搜索更准？"
  → LLM 编答案："提高检索精度可以通过混合检索、重排序、查询改写等方法..."
  → 用这个假答案去搜 → 命中"检索精度优化"相关文档
```

### 8.3 多查询扩展：一个问题变三个

把一个原始问题改写成多个不同角度的查询，分别检索，合并去重。覆盖面比单个查询大得多：

```
原始："RAG 系统有哪些常见问题？"
  → 改写 1："RAG 检索增强生成的局限性"
  → 改写 2："RAG 系统常见的失败模式"
  → 改写 3："如何诊断 RAG 系统的质量问题"
```

### 8.4 Step-back：退一步海阔天空

有些问题太具体，直接搜不出好结果。先问一个更抽象的"背景问题"，把背景信息和具体信息合并：

```
具体："Cross-Encoder 比 Bi-Encoder 慢多少？"
  → 退一步："Cross-Encoder 和 Bi-Encoder 的架构区别是什么？"
  → 两个问题分别检索，合并结果
```

---

## 9. 不评估就是在瞎调

### 9.1 调参的幻觉

调 chunk_size 从 200 到 300，感觉回答变好了？很可能只是错觉。没有评估的调参就是玄学——你以为在优化，实际上可能在倒退。

RAGAS 定义了四个正交的评估维度，是目前最常用的 RAG 评估方案：

### 9.2 四个指标一句话总结

| 指标 | 问什么 | 低了说明 |
|------|--------|---------|
| Faithfulness | 回答里的每句话，文档里能找到依据吗？ | 模型在编 |
| Answer Relevancy | 回答切题吗？ | 跑题了 |
| Context Precision | 搜回来的文档，有多少是真正有用的？ | 检索噪声大 |
| Context Recall | 回答需要的信息，都搜到了吗？ | 漏了关键信息 |

Faithfulness 是最重要的指标。它的计算方式很直接：从回答中拆出所有"陈述"，逐一检查是否能从检索到的文档中找到支撑。找不到支撑的陈述 = 幻觉。

### 9.3 诊断 → 处方

```
Faithfulness 低     → Prompt 强调"只基于文档回答"，降低 temperature
Answer Relevancy 低 → 检查检索结果是否跑偏，改进 Prompt 模板
Context Precision 低 → 加 Cross-Encoder 重排序，减小 final_k
Context Recall 低    → 增大 first_stage_k，用混合检索 + 查询改写
```

---

## 10. 一张图收尾

```
RAG 管道全景
═══════════════

离线索引
├── 文档加载（PDF / 网页 / 文本 → 清洗）
├── 文本分块（递归字符最通用，语义分块质量最高）
├── Embedding（sentence-transformers 本地跑 / API 调用）
└── 向量库（ChromaDB 够用，Milvus/Pinecone 上生产）

在线查询
├── 查询改写（HyDE / 多查询 / Step-back）
├── 混合检索（BM25 + 向量 + RRF 融合）
├── Cross-Encoder 重排序（精排 Top-K）
└── LLM 生成（基于精选文档回答）

评估
└── RAGAS（Faithfulness 是底线，四个指标一起看）
```

Phase 2 的目标是把 RAG 管道从头到尾搞明白。有了这个基础，Phase 3 进入框架实战（LangChain、LangGraph、CrewAI），RAG 会作为 Agent 的一个能力模块被集成进去，而不是孤立的管道。
