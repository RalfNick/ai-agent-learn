# Phase6 02 Knowledge Ingestion

这一节把 Phase6 从 API 骨架推进到“有资料可检索”。

它仍然不调用 LLM，也不接 LangGraph。目标是先建立一个稳定的数据面：

- 加载 Markdown / TXT / PDF 文档
- 切分成可检索 chunk
- 建立本地 JSON index
- 用确定性的 hybrid retrieval 查回相关片段

## 为什么先做本地 index

Phase6 后面会接 Chroma / Milvus / rerank / LangGraph，但第一版不应该一上来堆生产组件。

这里先做一个 stdlib 为主的本地索引，原因是：

- 测试稳定，不依赖外部 embedding API。
- 检索接口清楚，后面换向量库时边界不变。
- 可以快速验证 sources、snippet、score、path 这些字段是否够用。

## 目录结构

```text
02-knowledge-ingestion/
├── knowledge/
│   ├── loaders.py      # 文档加载
│   ├── chunking.py     # Markdown/text chunk
│   ├── index.py        # 本地 hybrid index
│   └── models.py       # Document/Chunk/RetrievalResult
├── ingest.py           # 构建 index
├── search.py           # 查询 index
└── tests/
```

## 安装

```bash
cd phase-6-capstone/02-knowledge-ingestion
python3 -m pip install -r requirements.txt
```

`pypdf` 只用于 PDF 文本抽取。只处理 Markdown 时不需要外部服务。

## 构建索引

从仓库根目录运行：

```bash
python3 phase-6-capstone/02-knowledge-ingestion/ingest.py \
  --source docs/phase-6 \
  --index /tmp/phase6-knowledge-index.json
```

示例输出：

```json
{
  "index_path": "/tmp/phase6-knowledge-index.json",
  "stats": {
    "document_count": 3,
    "chunk_count": 9,
    "token_count": 1200
  }
}
```

## 查询索引

```bash
python3 phase-6-capstone/02-knowledge-ingestion/search.py \
  --index /tmp/phase6-knowledge-index.json \
  --query "企业知识库 Agent trace" \
  --limit 3
```

返回字段包括：

- `title`
- `path`
- `score`
- `lexical_score`
- `vector_score`
- `snippet`

## 当前 retrieval 策略

当前是一个轻量 hybrid retrieval：

```text
final_score = 0.65 * lexical_score + 0.35 * vector_score
```

其中：

- lexical score：query token 和 chunk token 的重叠。
- vector score：稳定哈希 term-frequency 向量的余弦相似度。
- 中文文本会额外生成单字和 bigram token，避免 `知识库`、`检索` 这类短词完全召回不到。

这不是 Phase2 的最终最优配置，也不替代 rerank。它的价值是让 Capstone 的知识数据面先可运行、可测试、可持久化。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-6-capstone/02-knowledge-ingestion/tests
```

测试覆盖：

- Markdown metadata 加载
- chunk source metadata 保留
- hybrid retrieval 排名
- index save/load 后检索仍可用

## 下一步

`03-agentic-qa-runtime` 会复用这里的 retrieval contract：

```python
results = index.search(question, limit=5)
```

然后把结果交给 LangGraph workflow 做 context grading、answer generation、faithfulness check、repair 或 abstain。
