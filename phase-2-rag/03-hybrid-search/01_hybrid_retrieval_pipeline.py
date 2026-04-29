"""
01_hybrid_retrieval_pipeline.py — 混合检索管道：BM25 + 向量 + RRF 融合

学习目标：
1. 理解中文场景下 BM25 需要分词（jieba）才能发挥作用
2. 掌握 BM25 稀疏检索 + 向量稠密检索的互补关系
3. 实现 RRF（Reciprocal Rank Fusion）排序融合
4. 通过对比实验直观感受三种检索方式的差异

架构：
    用户查询
       │
       ├──────────────────┐
       ▼                  ▼
    ┌──────────┐    ┌──────────┐
    │  BM25    │    │  向量    │
    │ 稀疏检索  │    │ 稠密检索  │
    │ (关键词)  │    │ (语义)   │
    └────┬─────┘    └────┬─────┘
         │               │
         └───────┬───────┘
                 ▼
          ┌──────────────┐
          │  RRF 融合排序  │
          └──────┬───────┘
                 ▼
           混合检索结果

运行方式：
    pip install -r requirements.txt
    python 01_hybrid_retrieval_pipeline.py
"""

from __future__ import annotations

import time

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer

console = Console()

# ============================================================
# 知识库文档（中文 RAG 领域）
# ============================================================

DOCUMENTS = [
    "RAG（Retrieval-Augmented Generation）是检索增强生成技术，通过检索外部知识库来增强大语言模型的回答质量，有效解决知识过时和幻觉问题。",
    "BM25 是一种经典的信息检索算法，基于词频（TF）和逆文档频率（IDF）来计算文档与查询的相关性，属于稀疏检索方法。",
    "向量数据库使用高维向量表示文本语义，通过余弦相似度或欧氏距离来衡量文本间的语义相关性，是稠密检索的核心组件。",
    "文本分块策略直接影响 RAG 系统的检索质量。常见方法包括固定大小分块、递归字符分块和基于语义的分块。",
    "Embedding 模型将自然语言文本映射到稠密向量空间，使得语义相近的文本在向量空间中距离更近。中文推荐 BGE、M3E 等模型。",
    "混合检索结合了稀疏检索（如 BM25）和稠密检索（如向量搜索）的优势，能够同时捕获精确的关键词匹配和模糊的语义匹配。",
    "HNSW（Hierarchical Navigable Small World）是一种高效的近似最近邻搜索算法，被 Chroma、Milvus 等向量数据库广泛采用。",
    "Cross-Encoder 重排序模型同时接收查询和文档作为输入对，输出精确的相关性分数，精度远高于 Bi-Encoder 但速度较慢。",
    "查询改写技术通过重新表述用户问题来提升检索效果，常见方法包括 HyDE（假设性文档嵌入）和多查询扩展。",
    "RAGAS 是一个 RAG 系统评估框架，提供了忠实度（Faithfulness）、答案相关性、上下文精确度和上下文召回率等评估指标。",
    "Reciprocal Rank Fusion（RRF）是一种排序融合算法，通过倒数排名加权将多个检索器的结果合并，不需要对分数做归一化。",
    "大语言模型的上下文窗口有限，RAG 系统需要通过检索和重排序筛选最相关的文档片段，避免信息过载导致回答质量下降。",
]

DOC_IDS = [f"doc_{i:02d}" for i in range(len(DOCUMENTS))]


# ============================================================
# 1. BM25 稀疏检索器（中文分词版）
# ============================================================

class BM25Retriever:
    """
    基于 jieba 分词的中文 BM25 检索器。

    与 02-advanced-rag/05_hybrid_search.py 中按字符切分不同，
    这里使用 jieba 分词，对中文关键词匹配效果更好。
    """

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ============================================================
# 2. 向量稠密检索器
# ============================================================

class DenseRetriever:
    """基于 sentence-transformers 的向量检索器"""

    def __init__(self, documents: list[str], model_name: str = "all-MiniLM-L6-v2") -> None:
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(documents, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.doc_embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


# ============================================================
# 3. RRF 排序融合
# ============================================================

def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF)

    公式：RRF_score(d) = Σ 1 / (k + rank_i(d))

    k 是平滑参数（通常取 60），防止排名靠前的文档权重过大。
    优点：不需要对不同检索器的分数做归一化，直接用排名融合。
    """
    fused_scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_idx, _score) in enumerate(ranking):
            if doc_idx not in fused_scores:
                fused_scores[doc_idx] = 0.0
            fused_scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# 4. 混合检索器
# ============================================================

class HybridRetriever:
    """融合 BM25 和向量检索的混合检索器"""

    def __init__(self, documents: list[str], rrf_k: int = 60) -> None:
        self.documents = documents
        self.rrf_k = rrf_k
        console.print("  [dim]初始化 BM25 检索器（jieba 分词）...[/dim]")
        self.bm25 = BM25Retriever(documents)
        console.print("  [dim]初始化向量检索器（all-MiniLM-L6-v2）...[/dim]")
        self.dense = DenseRetriever(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_multiplier: int = 2,
    ) -> list[tuple[int, float, str]]:
        """
        混合检索流程：
        1. BM25 和向量检索各取 top_k * multiplier 个候选
        2. RRF 融合排序
        3. 返回 top_k 结果
        """
        n_candidates = top_k * candidate_multiplier
        bm25_results = self.bm25.search(query, top_k=n_candidates)
        dense_results = self.dense.search(query, top_k=n_candidates)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=self.rrf_k)
        return [
            (doc_idx, score, self.documents[doc_idx])
            for doc_idx, score in fused[:top_k]
        ]


# ============================================================
# 5. 对比实验
# ============================================================

def compare_retrieval_methods(query: str, retriever: HybridRetriever) -> None:
    console.print(f"\n[bold yellow]查询：{query}[/bold yellow]\n")

    bm25_results = retriever.bm25.search(query, top_k=3)
    dense_results = retriever.dense.search(query, top_k=3)
    hybrid_results = retriever.search(query, top_k=3)

    methods: list[tuple[str, list, str]] = [
        ("BM25 稀疏检索", bm25_results, "red"),
        ("向量稠密检索", dense_results, "blue"),
    ]

    for name, results, color in methods:
        table = Table(title=name, show_header=True)
        table.add_column("排名", style="white", width=4)
        table.add_column("分数", style=color, width=10)
        table.add_column("文档", style="white", max_width=70)

        for rank, (idx, score) in enumerate(results):
            table.add_row(str(rank + 1), f"{score:.4f}", DOCUMENTS[idx][:68] + "...")
        console.print(table)

    table = Table(title="混合检索 (RRF)", show_header=True)
    table.add_column("排名", style="white", width=4)
    table.add_column("RRF 分数", style="green", width=10)
    table.add_column("文档", style="white", max_width=70)

    for rank, (idx, score, _doc) in enumerate(hybrid_results):
        table.add_row(str(rank + 1), f"{score:.6f}", DOCUMENTS[idx][:68] + "...")
    console.print(table)


def run_timing_benchmark(retriever: HybridRetriever) -> None:
    console.print(Panel("性能基准测试", style="bold cyan"))

    queries = ["BM25 算法原理", "语义搜索", "RAG 评估指标"]
    methods = {
        "BM25": lambda q: retriever.bm25.search(q, top_k=3),
        "Dense": lambda q: retriever.dense.search(q, top_k=3),
        "Hybrid": lambda q: retriever.search(q, top_k=3),
    }

    table = Table(title="检索延迟对比 (ms)", show_header=True)
    table.add_column("查询", style="white", max_width=20)
    for name in methods:
        table.add_column(name, style="cyan", width=10)

    for query in queries:
        row = [query[:18]]
        for _name, fn in methods.items():
            start = time.perf_counter()
            fn(query)
            elapsed_ms = (time.perf_counter() - start) * 1000
            row.append(f"{elapsed_ms:.1f}")
        table.add_row(*row)

    console.print(table)


# ============================================================
# 6. 入口
# ============================================================

if __name__ == "__main__":
    console.print(Panel("混合检索管道：BM25 + 向量 + RRF", style="bold blue"))

    console.print("[bold]初始化混合检索器...[/bold]")
    retriever = HybridRetriever(DOCUMENTS)

    test_queries = [
        "BM25 算法",
        "如何评估 RAG 系统的质量",
        "语义搜索和关键词搜索的区别",
        "向量数据库用什么搜索算法",
    ]

    for q in test_queries:
        compare_retrieval_methods(q, retriever)

    run_timing_benchmark(retriever)

    console.print("\n[dim]下一步 -> 02_rerank_pipeline.py 添加 Cross-Encoder 重排序[/dim]")
