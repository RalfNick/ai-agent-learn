"""
02_rerank_pipeline.py — 两阶段检索：混合检索 + Cross-Encoder 重排序

学习目标：
1. 理解"检索-重排序"两阶段架构的必要性
2. 将混合检索（BM25 + 向量 + RRF）作为第一阶段粗检索
3. 用 Cross-Encoder 作为第二阶段精排
4. 对比有无重排序的检索效果和延迟

两阶段架构：
    用户查询
       │
       ▼
    ┌──────────────────────────┐
    │ 第一阶段：混合粗检索       │
    │ BM25 + Dense + RRF       │
    │ 从 N 篇中选 ~20 个候选    │  速度快，召回率高
    └────────────┬─────────────┘
                 │ ~20 个候选
                 ▼
    ┌──────────────────────────┐
    │ 第二阶段：Cross-Encoder   │
    │ 对每个 (query, doc) 对    │
    │ 计算精确相关性分数         │  精度高，速度慢
    └────────────┬─────────────┘
                 │ Top-K 最相关
                 ▼
           送入 LLM 生成回答

运行方式：
    python 02_rerank_pipeline.py
"""

from __future__ import annotations

import time

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer

console = Console()

# ============================================================
# 知识库（复用 01 的文档集）
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


# ============================================================
# 1. 混合检索器（从 01 复用核心逻辑）
# ============================================================

class BM25Retriever:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class DenseRetriever:
    def __init__(self, documents: list[str], model_name: str = "all-MiniLM-L6-v2") -> None:
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(documents, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.doc_embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_idx, _) in enumerate(ranking):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        bm25_results = self.bm25.search(query, top_k=top_k)
        dense_results = self.dense.search(query, top_k=top_k)
        return reciprocal_rank_fusion([bm25_results, dense_results])[:top_k]


# ============================================================
# 2. 两阶段检索器：混合检索 + Cross-Encoder 重排序
# ============================================================

class TwoStageRetriever:
    """
    第一阶段：HybridRetriever 粗检索 first_stage_k 个候选
    第二阶段：CrossEncoder 对候选精排，返回 final_k 个结果
    """

    def __init__(
        self,
        documents: list[str],
        cross_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.documents = documents
        console.print("  [dim]初始化混合检索器...[/dim]")
        self.hybrid = HybridRetriever(documents)
        console.print("  [dim]加载 Cross-Encoder 重排序模型...[/dim]")
        self.cross_encoder = CrossEncoder(cross_model_name)

    def search(
        self,
        query: str,
        first_stage_k: int = 10,
        final_k: int = 3,
    ) -> list[tuple[int, float]]:
        candidates = self.hybrid.search(query, top_k=first_stage_k)
        candidate_indices = [idx for idx, _ in candidates]

        pairs = [[query, self.documents[idx]] for idx in candidate_indices]
        rerank_scores = self.cross_encoder.predict(pairs)

        reranked = sorted(
            zip(candidate_indices, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(int(idx), float(score)) for idx, score in reranked[:final_k]]


# ============================================================
# 3. 对比实验
# ============================================================

def compare_with_without_reranking(retriever: TwoStageRetriever) -> None:
    console.print(Panel("重排序效果对比", style="bold cyan"))

    queries = [
        "重排序模型如何工作？",
        "RAG 系统的评估方法有哪些？",
        "如何处理 LLM 上下文长度限制？",
        "BM25 和向量检索有什么区别？",
    ]

    for query in queries:
        console.print(f"\n[bold yellow]查询：{query}[/bold yellow]")

        # 无重排序：仅混合检索
        no_rerank = retriever.hybrid.search(query, top_k=3)

        # 有重排序：混合检索 + Cross-Encoder
        with_rerank = retriever.search(query, first_stage_k=8, final_k=3)

        table = Table(show_header=True)
        table.add_column("排名", width=4)
        table.add_column("仅混合检索", style="red", max_width=50)
        table.add_column("+ Cross-Encoder 重排序", style="green", max_width=50)

        for rank in range(min(3, len(no_rerank), len(with_rerank))):
            nr_idx, nr_score = no_rerank[rank]
            rr_idx, rr_score = with_rerank[rank]
            table.add_row(
                str(rank + 1),
                f"[{nr_score:.4f}] {DOCUMENTS[nr_idx][:42]}...",
                f"[{rr_score:.2f}] {DOCUMENTS[rr_idx][:42]}...",
            )
        console.print(table)


def run_latency_analysis(retriever: TwoStageRetriever) -> None:
    console.print(Panel("延迟 vs 精度分析", style="bold cyan"))

    query = "如何提升 RAG 系统的检索质量？"
    configs = [
        ("仅 BM25", lambda: retriever.hybrid.bm25.search(query, top_k=3)),
        ("仅向量", lambda: retriever.hybrid.dense.search(query, top_k=3)),
        ("混合 (RRF)", lambda: retriever.hybrid.search(query, top_k=3)),
        ("混合 + Rerank (k=5)", lambda: retriever.search(query, first_stage_k=5, final_k=3)),
        ("混合 + Rerank (k=10)", lambda: retriever.search(query, first_stage_k=10, final_k=3)),
    ]

    table = Table(title=f"查询：{query}", show_header=True)
    table.add_column("方法", style="cyan", width=24)
    table.add_column("延迟 (ms)", style="yellow", width=10)
    table.add_column("Top-1 结果", style="white", max_width=50)

    for name, fn in configs:
        start = time.perf_counter()
        results = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        top_idx = results[0][0] if results else -1
        top_doc = DOCUMENTS[top_idx][:48] + "..." if top_idx >= 0 else "N/A"
        table.add_row(name, f"{elapsed_ms:.1f}", top_doc)

    console.print(table)
    console.print("[dim]注意：first_stage_k 越大，Cross-Encoder 需要评估的对越多，延迟越高[/dim]")


# ============================================================
# 4. 入口
# ============================================================

if __name__ == "__main__":
    console.print(Panel("两阶段检索：混合检索 + Cross-Encoder 重排序", style="bold blue"))

    console.print("[bold]初始化两阶段检索器...[/bold]")
    retriever = TwoStageRetriever(DOCUMENTS)
    console.print()

    compare_with_without_reranking(retriever)
    run_latency_analysis(retriever)

    console.print("\n[dim]下一步 -> 03_full_rag_pipeline.py 添加查询改写和 LLM 生成[/dim]")
