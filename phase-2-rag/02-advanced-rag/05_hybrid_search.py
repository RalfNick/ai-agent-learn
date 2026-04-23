"""
05_hybrid_search.py — 混合检索：BM25 + 向量相似度

学习目标：
1. 理解纯向量检索的局限：语义匹配强但关键词匹配弱
2. 掌握 BM25 稀疏检索的原理和实现
3. 学会将 BM25 和向量检索融合（Hybrid Search）
4. 理解 RRF（Reciprocal Rank Fusion）排序融合算法

核心概念：
- 稀疏检索（BM25）：基于关键词匹配，擅长精确匹配
- 稠密检索（向量）：基于语义相似度，擅长模糊匹配
- 混合检索：两者互补，覆盖更多相关文档
- RRF：一种简单有效的排序融合方法

对比：
    查询："Python 装饰器"
    ┌─────────────────────────────────────────────────┐
    │ BM25 擅长：                                      │
    │   ✓ "Python 装饰器是一种语法糖..."  （精确匹配）    │
    │   ✗ "函数包装器模式可以..."          （语义相关）    │
    │                                                 │
    │ 向量检索擅长：                                    │
    │   ✓ "函数包装器模式可以..."          （语义匹配）    │
    │   ✗ "Python 装饰器是一种语法糖..."  （可能排名低）   │
    │                                                 │
    │ 混合检索：两者都能找到！                            │
    └─────────────────────────────────────────────────┘

运行方式：
    python 05_hybrid_search.py
"""

import os
import sys
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer

console = Console()


# ============================================================
# 准备测试数据
# ============================================================

DOCUMENTS = [
    "RAG（Retrieval-Augmented Generation）是检索增强生成技术，通过检索外部知识来增强 LLM 的回答质量。",
    "BM25 是一种经典的信息检索算法，基于词频和逆文档频率来计算文档与查询的相关性。",
    "向量数据库使用高维向量表示文本语义，通过余弦相似度或欧氏距离来衡量文本间的相关性。",
    "文本分块策略直接影响 RAG 系统的检索质量。常见的分块方法包括固定大小分块和递归字符分块。",
    "Embedding 模型将自然语言文本映射到稠密向量空间，使得语义相近的文本在向量空间中距离更近。",
    "混合检索结合了稀疏检索（如 BM25）和稠密检索（如向量搜索）的优势，能够同时捕获精确匹配和语义匹配。",
    "HNSW（Hierarchical Navigable Small World）是一种高效的近似最近邻搜索算法，被广泛用于向量数据库。",
    "Cross-Encoder 重排序模型可以对检索结果进行精细排序，显著提升检索精度，但计算成本较高。",
    "查询改写技术通过重新表述用户问题来提升检索效果，常见方法包括 HyDE 和多查询扩展。",
    "RAGAS 是一个 RAG 系统评估框架，提供了忠实度、答案相关性、上下文精确度等多个评估指标。",
]


# ============================================================
# 1. BM25 稀疏检索
# ============================================================

class BM25Retriever:
    """
    BM25 检索器。
    BM25 的核心思想：一个词在文档中出现越多（TF），在所有文档中出现越少（IDF），
    这个词对该文档的重要性就越高。
    """

    def __init__(self, documents: list[str]):
        self.documents = documents
        tokenized = [list(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """返回 (文档索引, BM25分数) 列表"""
        tokenized_query = list(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


# ============================================================
# 2. 向量稠密检索
# ============================================================

class DenseRetriever:
    """基于 sentence-transformers 的向量检索器"""

    def __init__(self, documents: list[str], model_name: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(documents, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """返回 (文档索引, 余弦相似度) 列表"""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.doc_embeddings, query_embedding.T).flatten()
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
    RRF 的优点：不需要对不同检索器的分数做归一化，直接用排名融合。
    """
    fused_scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, (doc_idx, _score) in enumerate(ranking):
            if doc_idx not in fused_scores:
                fused_scores[doc_idx] = 0.0
            fused_scores[doc_idx] += 1.0 / (k + rank + 1)

    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


# ============================================================
# 4. 混合检索器
# ============================================================

class HybridRetriever:
    """融合 BM25 和向量检索的混合检索器"""

    def __init__(self, documents: list[str]):
        self.documents = documents
        console.print("  初始化 BM25 检索器...")
        self.bm25 = BM25Retriever(documents)
        console.print("  初始化向量检索器...")
        self.dense = DenseRetriever(documents)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float, str]]:
        """
        混合检索：
        1. 分别用 BM25 和向量检索获取候选
        2. 用 RRF 融合排序
        3. 返回 top_k 结果
        """
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        dense_results = self.dense.search(query, top_k=top_k * 2)

        fused = reciprocal_rank_fusion([bm25_results, dense_results])

        results = []
        for doc_idx, score in fused[:top_k]:
            results.append((doc_idx, score, self.documents[doc_idx]))

        return results


# ============================================================
# 5. 对比实验
# ============================================================

def compare_retrieval_methods(query: str, retriever: HybridRetriever):
    """对比三种检索方法的结果"""
    console.print(f"\n[bold yellow]查询：{query}[/bold yellow]\n")

    bm25_results = retriever.bm25.search(query, top_k=3)
    dense_results = retriever.dense.search(query, top_k=3)
    hybrid_results = retriever.search(query, top_k=3)

    methods = [
        ("BM25 稀疏检索", bm25_results, "red"),
        ("向量稠密检索", dense_results, "blue"),
    ]

    for name, results, color in methods:
        table = Table(title=name, show_header=True)
        table.add_column("排名", style="white", width=4)
        table.add_column("分数", style=color, width=8)
        table.add_column("文档", style="white", max_width=60)

        for rank, (idx, score) in enumerate(results):
            table.add_row(str(rank + 1), f"{score:.4f}", DOCUMENTS[idx][:60] + "...")

        console.print(table)

    table = Table(title="🔀 混合检索 (RRF)", show_header=True)
    table.add_column("排名", style="white", width=4)
    table.add_column("RRF分数", style="green", width=8)
    table.add_column("文档", style="white", max_width=60)

    for rank, (idx, score, _doc) in enumerate(hybrid_results):
        table.add_row(str(rank + 1), f"{score:.4f}", DOCUMENTS[idx][:60] + "...")

    console.print(table)


# ============================================================
# 6. 演示
# ============================================================

if __name__ == "__main__":
    console.print(Panel("🔀 混合检索：BM25 + 向量", style="bold blue"))

    console.print("[bold]初始化检索器...[/bold]")
    retriever = HybridRetriever(DOCUMENTS)
    console.print()

    queries = [
        "BM25 算法",
        "如何评估 RAG 系统的质量",
        "语义搜索和关键词搜索的区别",
    ]

    for query in queries:
        compare_retrieval_methods(query, retriever)

    console.print("\n[dim]下一步 → 06_reranking.py 学习重排序技术[/dim]")
