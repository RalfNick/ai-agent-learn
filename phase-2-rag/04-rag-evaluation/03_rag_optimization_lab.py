"""
03_rag_optimization_lab.py — RAG 参数调优实验室

学习目标：
1. 理解 RAG 系统中可调参数及其相互影响
2. 实现参数扫描（parameter sweep）自动寻找最优配置
3. 用评估指标驱动优化决策
4. 掌握 Precision-Recall 权衡和延迟-质量权衡

可调参数：
    ┌─────────────────────────────────────────────┐
    │  检索阶段                                    │
    │  ├─ first_stage_k    粗检索候选数 (5~20)     │
    │  ├─ rrf_k            RRF 平滑参数 (10~100)   │
    │  └─ retriever_type   BM25/Dense/Hybrid       │
    │                                             │
    │  重排序阶段                                   │
    │  ├─ rerank_enabled   是否启用重排序           │
    │  └─ final_k          精排后保留数 (1~5)       │
    │                                             │
    │  生成阶段                                    │
    │  ├─ temperature      LLM 温度 (0.0~1.0)     │
    │  └─ prompt_template  Prompt 模板             │
    └─────────────────────────────────────────────┘

运行方式：
    cp .env.example .env  # 填入 API key
    python 03_rag_optimization_lab.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

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


def call_llm(prompt: str, temperature: float = 0.3) -> str:
    response = litellm.completion(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# 1. 知识库 + 评估数据集（带 ground truth）
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

EVAL_SET = [
    {
        "question": "BM25 和向量检索有什么区别？",
        "relevant_doc_indices": [1, 2, 5],
        "ground_truth": "BM25 基于关键词匹配（稀疏检索），向量检索基于语义相似度（稠密检索），两者互补。",
    },
    {
        "question": "Cross-Encoder 重排序的原理是什么？",
        "relevant_doc_indices": [7, 11],
        "ground_truth": "Cross-Encoder 同时编码查询和文档对，输出相关性分数，精度高但速度慢。",
    },
    {
        "question": "如何评估 RAG 系统的质量？",
        "relevant_doc_indices": [9],
        "ground_truth": "使用 RAGAS 框架的四个指标：忠实度、答案相关性、上下文精确度、上下文召回率。",
    },
    {
        "question": "混合检索有什么优势？",
        "relevant_doc_indices": [1, 2, 5, 10],
        "ground_truth": "混合检索结合 BM25 关键词匹配和向量语义匹配，通过 RRF 融合，兼顾精确和模糊匹配。",
    },
]


# ============================================================
# 2. 检索组件
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
# 3. 可配置 RAG 管道
# ============================================================

@dataclass
class RAGConfig:
    """RAG 管道的可调参数"""
    name: str
    retriever_type: str = "hybrid"       # "bm25" | "dense" | "hybrid"
    first_stage_k: int = 10
    rrf_k: int = 60
    rerank_enabled: bool = True
    final_k: int = 3
    temperature: float = 0.3
    prompt_style: str = "strict"         # "strict" | "flexible"


PROMPT_TEMPLATES = {
    "strict": """基于以下参考资料回答用户问题。
要求：只基于提供的参考资料回答，不要编造信息。如果资料不足，请说明。

参考资料：
{context}

问题：{question}

回答：""",
    "flexible": """参考以下资料回答问题。可以适当补充你的知识，但核心信息应来自参考资料。

参考资料：
{context}

问题：{question}

回答：""",
}


class ConfigurableRAGPipeline:
    """支持参数配置的 RAG 管道"""

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents)
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _retrieve(self, query: str, config: RAGConfig) -> list[tuple[int, float]]:
        k = config.first_stage_k
        if config.retriever_type == "bm25":
            return self.bm25.search(query, top_k=k)
        elif config.retriever_type == "dense":
            return self.dense.search(query, top_k=k)
        else:
            bm25_results = self.bm25.search(query, top_k=k)
            dense_results = self.dense.search(query, top_k=k)
            return reciprocal_rank_fusion(
                [bm25_results, dense_results], k=config.rrf_k
            )[:k]

    def _rerank(
        self, query: str, candidates: list[tuple[int, float]], config: RAGConfig
    ) -> list[tuple[int, float]]:
        if not config.rerank_enabled or not candidates:
            return candidates[: config.final_k]
        indices = [idx for idx, _ in candidates]
        pairs = [[query, self.documents[idx]] for idx in indices]
        scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(s)) for idx, s in reranked[: config.final_k]]

    def run(self, question: str, config: RAGConfig) -> dict:
        t0 = time.perf_counter()
        candidates = self._retrieve(question, config)
        t_retrieve = time.perf_counter() - t0

        t0 = time.perf_counter()
        final_docs = self._rerank(question, candidates, config)
        t_rerank = time.perf_counter() - t0

        context_parts = []
        for i, (idx, score) in enumerate(final_docs):
            context_parts.append(f"[来源 {i+1}] {self.documents[idx]}")
        context = "\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATES[config.prompt_style].format(
            context=context, question=question
        )

        t0 = time.perf_counter()
        answer = call_llm(prompt, temperature=config.temperature)
        t_generate = time.perf_counter() - t0

        return {
            "question": question,
            "answer": answer,
            "retrieved_indices": [idx for idx, _ in final_docs],
            "context": context,
            "timings": {
                "retrieve_ms": t_retrieve * 1000,
                "rerank_ms": t_rerank * 1000,
                "generate_ms": t_generate * 1000,
            },
        }


# ============================================================
# 4. 评估指标（轻量版，用于参数扫描）
# ============================================================

def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(top_k) if top_k else 0.0


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant) if relevant else 1.0


def mrr(retrieved: list[int], relevant: set[int]) -> float:
    for i, doc_idx in enumerate(retrieved):
        if doc_idx in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_faithfulness_quick(answer: str, contexts: list[str]) -> float:
    """轻量版忠实度：用 LLM 直接打分（0-1）"""
    context_text = "\n".join(contexts)
    prompt = f"""请评估以下回答对参考资料的忠实程度。
只输出一个 0 到 1 之间的数字（保留两位小数）。
1.0 = 完全基于参考资料，0.0 = 完全编造。

参考资料：{context_text}

回答：{answer}

忠实度分数："""
    result = call_llm(prompt)
    try:
        return max(0.0, min(1.0, float(result.strip())))
    except ValueError:
        return 0.5


def evaluate_config(
    pipeline: ConfigurableRAGPipeline,
    config: RAGConfig,
    eval_set: list[dict],
    with_faithfulness: bool = False,
) -> dict:
    """对一种配置运行评估"""
    p_scores, r_scores, mrr_scores = [], [], []
    faith_scores = []
    total_latency = 0.0

    for item in eval_set:
        relevant = set(item["relevant_doc_indices"])
        result = pipeline.run(item["question"], config)
        retrieved = result["retrieved_indices"]

        p_scores.append(precision_at_k(retrieved, relevant, config.final_k))
        r_scores.append(recall_at_k(retrieved, relevant, config.final_k))
        mrr_scores.append(mrr(retrieved, relevant))
        total_latency += sum(result["timings"].values())

        if with_faithfulness:
            faith = evaluate_faithfulness_quick(result["answer"], [result["context"]])
            faith_scores.append(faith)

    scores = {
        "precision": float(np.mean(p_scores)),
        "recall": float(np.mean(r_scores)),
        "mrr": float(np.mean(mrr_scores)),
        "avg_latency_ms": total_latency / len(eval_set),
    }
    if with_faithfulness:
        scores["faithfulness"] = float(np.mean(faith_scores))
    return scores


# ============================================================
# 5. 实验 1：检索策略对比
# ============================================================

def experiment_retriever_type(pipeline: ConfigurableRAGPipeline) -> None:
    console.print(Panel("实验 1：检索策略对比", style="bold cyan"))

    configs = [
        RAGConfig(name="BM25", retriever_type="bm25", rerank_enabled=False, final_k=3),
        RAGConfig(name="Dense", retriever_type="dense", rerank_enabled=False, final_k=3),
        RAGConfig(name="Hybrid", retriever_type="hybrid", rerank_enabled=False, final_k=3),
        RAGConfig(name="Hybrid+Rerank", retriever_type="hybrid", rerank_enabled=True, final_k=3),
    ]

    table = Table(title="检索策略 × 评估指标", show_header=True)
    table.add_column("策略", style="cyan", width=16)
    table.add_column("Precision", width=10)
    table.add_column("Recall", width=10)
    table.add_column("MRR", style="green", width=10)
    table.add_column("Faith.", style="yellow", width=10)
    table.add_column("延迟(ms)", style="dim", width=10)

    for cfg in configs:
        console.print(f"  [dim]测试: {cfg.name}...[/dim]")
        scores = evaluate_config(pipeline, cfg, EVAL_SET, with_faithfulness=True)
        table.add_row(
            cfg.name,
            f"{scores['precision']:.3f}",
            f"{scores['recall']:.3f}",
            f"{scores['mrr']:.3f}",
            f"{scores.get('faithfulness', 0):.3f}",
            f"{scores['avg_latency_ms']:.0f}",
        )

    console.print(table)


# ============================================================
# 6. 实验 2：first_stage_k × final_k 参数扫描
# ============================================================

def experiment_k_sweep(pipeline: ConfigurableRAGPipeline) -> None:
    console.print(Panel("实验 2：first_stage_k × final_k 参数扫描", style="bold cyan"))

    table = Table(title="参数扫描（Hybrid + Rerank）", show_header=True)
    table.add_column("first_k", style="cyan", width=8)
    table.add_column("final_k", style="cyan", width=8)
    table.add_column("Precision", width=10)
    table.add_column("Recall", width=10)
    table.add_column("MRR", style="green", width=10)
    table.add_column("延迟(ms)", style="dim", width=10)

    for first_k in [5, 8, 10]:
        for final_k in [1, 3, 5]:
            if final_k > first_k:
                continue
            cfg = RAGConfig(
                name=f"fk={first_k},rk={final_k}",
                retriever_type="hybrid",
                first_stage_k=first_k,
                rerank_enabled=True,
                final_k=final_k,
            )
            console.print(f"  [dim]first_k={first_k}, final_k={final_k}...[/dim]")
            scores = evaluate_config(pipeline, cfg, EVAL_SET)
            table.add_row(
                str(first_k), str(final_k),
                f"{scores['precision']:.3f}",
                f"{scores['recall']:.3f}",
                f"{scores['mrr']:.3f}",
                f"{scores['avg_latency_ms']:.0f}",
            )

    console.print(table)
    console.print("[dim]观察：final_k 越小 → Precision 越高但 Recall 越低[/dim]")
    console.print("[dim]观察：first_stage_k 越大 → Recall 越高但延迟越大[/dim]")


# ============================================================
# 7. 实验 3：Prompt 模板 × Temperature 对比
# ============================================================

def experiment_generation_params(pipeline: ConfigurableRAGPipeline) -> None:
    console.print(Panel("实验 3：Prompt 模板 × Temperature 对比", style="bold cyan"))

    table = Table(title="生成参数对比（Hybrid + Rerank, final_k=3）", show_header=True)
    table.add_column("Prompt", style="cyan", width=10)
    table.add_column("Temp", style="cyan", width=6)
    table.add_column("Faith.", style="green", width=10)
    table.add_column("示例回答", style="white", max_width=50)

    question = "BM25 和向量检索有什么区别？"

    for prompt_style in ["strict", "flexible"]:
        for temp in [0.1, 0.5, 0.8]:
            cfg = RAGConfig(
                name=f"{prompt_style}_t{temp}",
                retriever_type="hybrid",
                rerank_enabled=True,
                final_k=3,
                temperature=temp,
                prompt_style=prompt_style,
            )
            result = pipeline.run(question, cfg)
            faith = evaluate_faithfulness_quick(result["answer"], [result["context"]])
            table.add_row(
                prompt_style,
                f"{temp}",
                f"{faith:.2f}",
                result["answer"][:48] + "...",
            )

    console.print(table)
    console.print("[dim]观察：strict prompt + 低 temperature → 忠实度更高[/dim]")


# ============================================================
# 8. 入口
# ============================================================

if __name__ == "__main__":
    console.print(Panel("RAG 参数调优实验室", style="bold blue"))

    console.print("[bold]初始化可配置 RAG 管道...[/bold]")
    console.print("  [dim]BM25 + Dense + Cross-Encoder...[/dim]")
    pipeline = ConfigurableRAGPipeline(DOCUMENTS)
    console.print()

    experiment_retriever_type(pipeline)
    console.print()

    experiment_k_sweep(pipeline)
    console.print()

    experiment_generation_params(pipeline)

    console.print("\n[bold]调优总结：[/bold]")
    console.print("  1. Hybrid + Rerank 通常是最佳检索策略")
    console.print("  2. first_stage_k=8~10, final_k=3 是常见的平衡点")
    console.print("  3. strict prompt + temperature=0.1~0.3 → 高忠实度")
    console.print("  4. 用评估指标驱动每次调参决策，避免凭感觉优化")

    console.print("\n[dim]Phase 2 RAG 评估模块完成！[/dim]")
