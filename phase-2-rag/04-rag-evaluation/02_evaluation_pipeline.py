"""
02_evaluation_pipeline.py — 自动化评估管道：对比不同 RAG 检索配置

学习目标：
1. 构建可复用的评估管道，自动对比多种检索策略
2. 使用 embedding 相似度作为轻量级代理指标（无需 LLM API）
3. 理解 Precision/Recall/MRR/NDCG 等经典 IR 指标
4. 通过 A/B 测试找到最优检索配置

评估架构：
    评估数据集（问题 + 标注相关文档）
       │
       ▼
    ┌──────────────────────────────┐
    │  对每种检索配置运行检索        │
    │  BM25 / Dense / Hybrid / +RR │
    └──────────┬───────────────────┘
               │ 每种配置的检索结果
               ▼
    ┌──────────────────────────────┐
    │  计算 IR 指标                 │
    │  Precision@K / Recall@K      │
    │  MRR / NDCG                  │
    └──────────┬───────────────────┘
               │
               ▼
         对比表格 + 分析

运行方式（无需 API key）：
    python 02_evaluation_pipeline.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer

console = Console()


# ============================================================
# 1. 知识库 + 评估数据集
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
    "HyDE（Hypothetical Document Embeddings）先让 LLM 生成假设性答案文档，再用这个假设文档的向量去检索真实文档。",
    "多查询扩展（Multi-Query Expansion）将用户的一个问题改写为多个不同角度的查询，并行检索后合并去重，提高召回率。",
    "Step-back Prompting 先将具体问题抽象为更基础的问题，获取背景知识后再回答原始问题，适合需要深层推理的场景。",
]

EVAL_QUERIES = [
    {
        "query": "BM25 算法原理",
        "relevant_doc_indices": [1, 5],
    },
    {
        "query": "向量数据库用什么搜索算法",
        "relevant_doc_indices": [2, 6],
    },
    {
        "query": "如何评估 RAG 系统",
        "relevant_doc_indices": [9],
    },
    {
        "query": "重排序模型如何工作",
        "relevant_doc_indices": [7, 11],
    },
    {
        "query": "查询改写有哪些方法",
        "relevant_doc_indices": [8, 12, 13, 14],
    },
    {
        "query": "混合检索的优势",
        "relevant_doc_indices": [1, 2, 5, 10],
    },
    {
        "query": "文本分块策略",
        "relevant_doc_indices": [3],
    },
    {
        "query": "RAG 解决什么问题",
        "relevant_doc_indices": [0, 11],
    },
]


# ============================================================
# 2. 检索器（复用 03-hybrid-search 的核心逻辑）
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
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever) -> None:
        self.bm25 = bm25
        self.dense = dense

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        dense_results = self.dense.search(query, top_k=top_k * 2)
        return reciprocal_rank_fusion([bm25_results, dense_results])[:top_k]


class TwoStageRetriever:
    def __init__(
        self, hybrid: HybridRetriever, documents: list[str],
        cross_encoder: CrossEncoder,
    ) -> None:
        self.hybrid = hybrid
        self.documents = documents
        self.cross_encoder = cross_encoder

    def search(
        self, query: str, first_stage_k: int = 10, final_k: int = 5
    ) -> list[tuple[int, float]]:
        candidates = self.hybrid.search(query, top_k=first_stage_k)
        indices = [idx for idx, _ in candidates]
        pairs = [[query, self.documents[idx]] for idx in indices]
        scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(s)) for idx, s in reranked[:final_k]]


# ============================================================
# 3. 经典 IR 评估指标
# ============================================================

def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Precision@K：前 K 个结果中相关文档的比例"""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / len(top_k)


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Recall@K：前 K 个结果覆盖了多少相关文档"""
    top_k = retrieved[:k]
    if not relevant:
        return 1.0
    return len(set(top_k) & relevant) / len(relevant)


def mean_reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    """MRR：第一个相关文档出现在第几位（倒数）"""
    for i, doc_idx in enumerate(retrieved):
        if doc_idx in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """NDCG@K：归一化折损累积增益"""
    top_k = retrieved[:k]
    dcg = sum(
        (1.0 if doc_idx in relevant else 0.0) / np.log2(i + 2)
        for i, doc_idx in enumerate(top_k)
    )
    ideal = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class EvalResult:
    config_name: str
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    avg_latency_ms: float = 0.0


# ============================================================
# 4. 评估管道
# ============================================================

def evaluate_config(
    config_name: str,
    search_fn,
    eval_queries: list[dict],
    top_k: int = 5,
) -> EvalResult:
    """对一种检索配置运行完整评估"""
    p3_scores, p5_scores = [], []
    r3_scores, r5_scores = [], []
    mrr_scores, ndcg_scores = [], []
    latencies = []

    for item in eval_queries:
        query = item["query"]
        relevant = set(item["relevant_doc_indices"])

        start = time.perf_counter()
        results = search_fn(query, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)

        retrieved_indices = [idx for idx, _ in results]

        p3_scores.append(precision_at_k(retrieved_indices, relevant, 3))
        p5_scores.append(precision_at_k(retrieved_indices, relevant, 5))
        r3_scores.append(recall_at_k(retrieved_indices, relevant, 3))
        r5_scores.append(recall_at_k(retrieved_indices, relevant, 5))
        mrr_scores.append(mean_reciprocal_rank(retrieved_indices, relevant))
        ndcg_scores.append(ndcg_at_k(retrieved_indices, relevant, 5))

    return EvalResult(
        config_name=config_name,
        precision_at_3=float(np.mean(p3_scores)),
        precision_at_5=float(np.mean(p5_scores)),
        recall_at_3=float(np.mean(r3_scores)),
        recall_at_5=float(np.mean(r5_scores)),
        mrr=float(np.mean(mrr_scores)),
        ndcg_at_5=float(np.mean(ndcg_scores)),
        avg_latency_ms=float(np.mean(latencies)),
    )


# ============================================================
# 5. 运行对比实验
# ============================================================

def run_comparison() -> list[EvalResult]:
    console.print("[bold]初始化检索器...[/bold]")
    console.print("  [dim]BM25 (jieba)...[/dim]")
    bm25 = BM25Retriever(DOCUMENTS)
    console.print("  [dim]Dense (all-MiniLM-L6-v2)...[/dim]")
    dense = DenseRetriever(DOCUMENTS)
    hybrid = HybridRetriever(bm25, dense)
    console.print("  [dim]Cross-Encoder (ms-marco-MiniLM-L-6-v2)...[/dim]")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    two_stage = TwoStageRetriever(hybrid, DOCUMENTS, cross_encoder)
    console.print()

    configs = [
        ("BM25 Only", lambda q, top_k=5: bm25.search(q, top_k=top_k)),
        ("Dense Only", lambda q, top_k=5: dense.search(q, top_k=top_k)),
        ("Hybrid (RRF)", lambda q, top_k=5: hybrid.search(q, top_k=top_k)),
        ("Hybrid + Rerank", lambda q, top_k=5: two_stage.search(q, first_stage_k=10, final_k=top_k)),
    ]

    results = []
    for name, fn in configs:
        console.print(f"  [dim]评估: {name}...[/dim]")
        result = evaluate_config(name, fn, EVAL_QUERIES, top_k=5)
        results.append(result)

    return results


def display_comparison(results: list[EvalResult]) -> None:
    table = Table(title="检索配置 A/B 对比", show_header=True)
    table.add_column("配置", style="cyan", width=18)
    table.add_column("P@3", style="white", width=6)
    table.add_column("P@5", style="white", width=6)
    table.add_column("R@3", style="white", width=6)
    table.add_column("R@5", style="white", width=6)
    table.add_column("MRR", style="green", width=6)
    table.add_column("NDCG@5", style="green", width=7)
    table.add_column("延迟(ms)", style="yellow", width=9)

    best_mrr = max(r.mrr for r in results)
    best_ndcg = max(r.ndcg_at_5 for r in results)

    for r in results:
        mrr_str = f"[bold green]{r.mrr:.3f}[/bold green]" if r.mrr == best_mrr else f"{r.mrr:.3f}"
        ndcg_str = f"[bold green]{r.ndcg_at_5:.3f}[/bold green]" if r.ndcg_at_5 == best_ndcg else f"{r.ndcg_at_5:.3f}"
        table.add_row(
            r.config_name,
            f"{r.precision_at_3:.3f}",
            f"{r.precision_at_5:.3f}",
            f"{r.recall_at_3:.3f}",
            f"{r.recall_at_5:.3f}",
            mrr_str,
            ndcg_str,
            f"{r.avg_latency_ms:.1f}",
        )

    console.print(table)


def display_per_query_analysis(results: list[EvalResult]) -> None:
    """展示每个查询在不同配置下的 MRR"""
    console.print("[bold]初始化检索器（per-query 分析）...[/bold]")
    bm25 = BM25Retriever(DOCUMENTS)
    dense = DenseRetriever(DOCUMENTS)
    hybrid = HybridRetriever(bm25, dense)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    two_stage = TwoStageRetriever(hybrid, DOCUMENTS, cross_encoder)

    configs = [
        ("BM25", lambda q: bm25.search(q, top_k=5)),
        ("Dense", lambda q: dense.search(q, top_k=5)),
        ("Hybrid", lambda q: hybrid.search(q, top_k=5)),
        ("+Rerank", lambda q: two_stage.search(q, first_stage_k=10, final_k=5)),
    ]

    table = Table(title="Per-Query MRR 分析", show_header=True)
    table.add_column("查询", style="white", max_width=22)
    for name, _ in configs:
        table.add_column(name, style="cyan", width=8)

    for item in EVAL_QUERIES:
        relevant = set(item["relevant_doc_indices"])
        row = [item["query"][:20]]
        for _, fn in configs:
            retrieved = [idx for idx, _ in fn(item["query"])]
            mrr = mean_reciprocal_rank(retrieved, relevant)
            color = "green" if mrr >= 0.5 else "yellow" if mrr > 0 else "red"
            row.append(f"[{color}]{mrr:.2f}[/{color}]")
        table.add_row(*row)

    console.print(table)


# ============================================================
# 6. 入口
# ============================================================

if __name__ == "__main__":
    console.print(Panel("自动化评估管道：对比不同检索配置", style="bold blue"))

    results = run_comparison()
    console.print()
    display_comparison(results)

    console.print()
    display_per_query_analysis(results)

    console.print("\n[bold]指标说明：[/bold]")
    console.print("  P@K  = 前 K 个结果中相关文档的比例（精确度）")
    console.print("  R@K  = 前 K 个结果覆盖了多少相关文档（召回率）")
    console.print("  MRR  = 第一个相关文档的排名倒数（越高越好）")
    console.print("  NDCG = 归一化折损累积增益（考虑排名位置的综合指标）")

    console.print("\n[dim]下一步 → 03_rag_optimization_lab.py 参数调优实验室[/dim]")