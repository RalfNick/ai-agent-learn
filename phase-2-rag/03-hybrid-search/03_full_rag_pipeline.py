"""
03_full_rag_pipeline.py — 完整 RAG 管道：查询改写 + 混合检索 + 重排序 + LLM 生成

学习目标：
1. 将查询改写、混合检索、重排序、LLM 生成串联成完整管道
2. 理解每个阶段的作用和中间结果
3. 掌握 HyDE 和 Multi-Query 查询改写的实际效果
4. 实现带上下文引用的 RAG 回答生成

完整管道架构：
    用户问题
       │
       ▼
    ┌──────────────────┐
    │  查询改写         │  HyDE / Multi-Query / Step-back
    │  Query Transform │  扩大检索覆盖面
    └────────┬─────────┘
             │ 改写后的查询（1~N 个）
             ▼
    ┌──────────────────┐
    │  混合检索         │  BM25 + Dense + RRF
    │  Hybrid Search   │  粗筛候选文档
    └────────┬─────────┘
             │ ~20 个候选
             ▼
    ┌──────────────────┐
    │  Cross-Encoder   │  精排候选
    │  Reranking       │  选出 Top-K
    └────────┬─────────┘
             │ Top-K 最相关文档
             ▼
    ┌──────────────────┐
    │  LLM 生成回答     │  基于检索到的上下文
    │  Generation      │  附带引用来源
    └──────────────────┘

运行方式：
    cp .env.example .env  # 填入 API key
    python 03_full_rag_pipeline.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import jieba
import litellm
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

console = Console()

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")


# ============================================================
# 知识库文档
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
    "HyDE（Hypothetical Document Embeddings）先让 LLM 生成假设性答案文档，再用这个假设文档的向量去检索真实文档，弥合查询与文档之间的语义鸿沟。",
    "多查询扩展（Multi-Query Expansion）将用户的一个问题改写为多个不同角度的查询，并行检索后合并去重，提高召回率。",
    "Step-back Prompting 先将具体问题抽象为更基础的问题，获取背景知识后再回答原始问题，适合需要深层推理的场景。",
]


# ============================================================
# 1. LLM 调用工具
# ============================================================

def call_llm(prompt: str, temperature: float = 0.7) -> str:
    response = litellm.completion(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# 2. 检索组件（复用 01/02 的核心逻辑）
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


# ============================================================
# 3. 查询改写
# ============================================================

def hyde_transform(query: str) -> str:
    """HyDE：让 LLM 生成假设性答案，用答案去检索"""
    prompt = f"""请针对以下问题，写一段简短的假设性答案（约 100 字）。
不需要完全准确，只需要包含可能相关的关键概念和术语。

问题：{query}

假设性答案："""
    return call_llm(prompt)


def multi_query_transform(query: str, n: int = 3) -> list[str]:
    """将一个查询改写为多个不同角度的查询"""
    prompt = f"""请将以下用户问题改写为 {n} 个不同角度的搜索查询。
每个查询应该从不同的角度描述同一个信息需求。
每行一个查询，不要编号。

用户问题：{query}

改写后的查询："""
    result = call_llm(prompt)
    queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
    return queries[:n]


def stepback_transform(query: str) -> str:
    """Step-back：将具体问题转化为更抽象的背景问题"""
    prompt = f"""请将以下具体问题转化为一个更抽象、更基础的问题。
这个抽象问题应该能帮助获取回答原始问题所需的背景知识。

具体问题：{query}

抽象问题（一句话）："""
    return call_llm(prompt)


# ============================================================
# 4. 完整 RAG 管道
# ============================================================

@dataclass
class RAGResult:
    """RAG 管道的完整输出"""
    query: str
    transformed_queries: list[str]
    retrieved_docs: list[tuple[int, float]]
    reranked_docs: list[tuple[int, float]]
    context: str
    answer: str
    timings: dict[str, float] = field(default_factory=dict)


class FullRAGPipeline:
    """端到端 RAG 管道：查询改写 → 混合检索 → 重排序 → 生成"""

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        console.print("  [dim]初始化 BM25...[/dim]")
        self.bm25 = BM25Retriever(documents)
        console.print("  [dim]初始化向量检索器...[/dim]")
        self.dense = DenseRetriever(documents)
        console.print("  [dim]加载 Cross-Encoder...[/dim]")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _hybrid_search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        bm25_results = self.bm25.search(query, top_k=top_k)
        dense_results = self.dense.search(query, top_k=top_k)
        return reciprocal_rank_fusion([bm25_results, dense_results])[:top_k]

    def _rerank(self, query: str, candidates: list[tuple[int, float]], top_k: int = 3) -> list[tuple[int, float]]:
        if not candidates:
            return []
        indices = [idx for idx, _ in candidates]
        pairs = [[query, self.documents[idx]] for idx in indices]
        scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(s)) for idx, s in reranked[:top_k]]

    def _build_context(self, doc_indices: list[tuple[int, float]]) -> str:
        parts = []
        for i, (idx, score) in enumerate(doc_indices):
            parts.append(f"[来源 {i+1}] (相关度: {score:.2f})\n{self.documents[idx]}")
        return "\n\n".join(parts)

    def _generate_answer(self, query: str, context: str) -> str:
        prompt = f"""基于以下检索到的参考资料回答用户问题。
要求：
1. 只基于提供的参考资料回答，不要编造信息
2. 如果参考资料不足以回答，请明确说明
3. 在回答中标注引用来源（如 [来源 1]）

参考资料：
{context}

用户问题：{query}

回答："""
        return call_llm(prompt, temperature=0.3)

    def run(
        self,
        query: str,
        transform: str = "none",
        first_stage_k: int = 10,
        final_k: int = 3,
    ) -> RAGResult:
        """
        执行完整 RAG 管道。

        Args:
            query: 用户问题
            transform: 查询改写策略 ("none" | "hyde" | "multi_query" | "stepback")
            first_stage_k: 混合检索候选数
            final_k: 重排序后保留数
        """
        timings: dict[str, float] = {}

        # 阶段 1：查询改写
        t0 = time.perf_counter()
        if transform == "hyde":
            hypo_doc = hyde_transform(query)
            search_queries = [hypo_doc]
        elif transform == "multi_query":
            search_queries = multi_query_transform(query)
        elif transform == "stepback":
            sb_query = stepback_transform(query)
            search_queries = [query, sb_query]
        else:
            search_queries = [query]
        timings["query_transform"] = time.perf_counter() - t0

        # 阶段 2：混合检索（对每个改写查询检索，合并去重）
        t0 = time.perf_counter()
        all_candidates: dict[int, float] = {}
        for sq in search_queries:
            for idx, score in self._hybrid_search(sq, top_k=first_stage_k):
                if idx not in all_candidates or score > all_candidates[idx]:
                    all_candidates[idx] = score
        retrieved = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        timings["hybrid_search"] = time.perf_counter() - t0

        # 阶段 3：Cross-Encoder 重排序
        t0 = time.perf_counter()
        reranked = self._rerank(query, retrieved, top_k=final_k)
        timings["reranking"] = time.perf_counter() - t0

        # 阶段 4：构建上下文 + LLM 生成
        t0 = time.perf_counter()
        context = self._build_context(reranked)
        answer = self._generate_answer(query, context)
        timings["generation"] = time.perf_counter() - t0

        return RAGResult(
            query=query,
            transformed_queries=search_queries,
            retrieved_docs=retrieved,
            reranked_docs=reranked,
            context=context,
            answer=answer,
            timings=timings,
        )


# ============================================================
# 5. 可视化展示
# ============================================================

def display_result(result: RAGResult) -> None:
    console.print(f"\n[bold yellow]问题：{result.query}[/bold yellow]")

    if len(result.transformed_queries) > 1 or result.transformed_queries[0] != result.query:
        console.print("[bold]查询改写结果：[/bold]")
        for i, q in enumerate(result.transformed_queries):
            console.print(f"  {i+1}. {q[:100]}{'...' if len(q) > 100 else ''}")

    table = Table(title="混合检索候选 → 重排序结果", show_header=True)
    table.add_column("排名", width=4)
    table.add_column("Rerank 分数", style="green", width=12)
    table.add_column("文档", style="white", max_width=70)

    for rank, (idx, score) in enumerate(result.reranked_docs):
        table.add_row(str(rank + 1), f"{score:.4f}", DOCUMENTS[idx][:68] + "...")
    console.print(table)

    console.print(Panel(result.answer, title="RAG 回答", style="bold green"))

    timing_parts = [f"{k}: {v*1000:.0f}ms" for k, v in result.timings.items()]
    total = sum(result.timings.values()) * 1000
    console.print(f"[dim]耗时 | {' | '.join(timing_parts)} | 总计: {total:.0f}ms[/dim]")


# ============================================================
# 6. 对比不同查询改写策略
# ============================================================

def compare_transform_strategies(pipeline: FullRAGPipeline) -> None:
    console.print(Panel("查询改写策略对比", style="bold cyan"))

    query = "如何提升 RAG 系统的检索质量？"
    strategies = ["none", "hyde", "multi_query", "stepback"]
    strategy_names = {
        "none": "无改写",
        "hyde": "HyDE",
        "multi_query": "多查询扩展",
        "stepback": "Step-back",
    }

    table = Table(title=f"查询：{query}", show_header=True)
    table.add_column("策略", style="cyan", width=12)
    table.add_column("Top-1 文档", style="white", max_width=50)
    table.add_column("Rerank 分数", style="green", width=12)
    table.add_column("总耗时 (ms)", style="yellow", width=12)

    for strategy in strategies:
        console.print(f"  [dim]测试策略: {strategy_names[strategy]}...[/dim]")
        result = pipeline.run(query, transform=strategy, final_k=3)
        if result.reranked_docs:
            top_idx, top_score = result.reranked_docs[0]
            top_doc = DOCUMENTS[top_idx][:48] + "..."
        else:
            top_score, top_doc = 0.0, "N/A"
        total_ms = sum(result.timings.values()) * 1000
        table.add_row(strategy_names[strategy], top_doc, f"{top_score:.4f}", f"{total_ms:.0f}")

    console.print(table)


# ============================================================
# 7. 入口
# ============================================================

if __name__ == "__main__":
    console.print(Panel("完整 RAG 管道：查询改写 + 混合检索 + 重排序 + 生成", style="bold blue"))

    console.print("[bold]初始化 RAG 管道...[/bold]")
    pipeline = FullRAGPipeline(DOCUMENTS)
    console.print()

    demo_queries = [
        ("BM25 和向量检索有什么区别？", "none"),
        ("怎么让检索结果更准确？", "hyde"),
        ("RAG 系统有哪些评估指标？", "multi_query"),
    ]

    for query, transform in demo_queries:
        result = pipeline.run(query, transform=transform, final_k=3)
        display_result(result)

    compare_transform_strategies(pipeline)

    console.print("\n[dim]Phase 2 混合检索 + Rerank 练习完成！[/dim]")
