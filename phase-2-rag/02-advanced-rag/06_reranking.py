"""
06_reranking.py — 重排序：Cross-Encoder 精排

学习目标：
1. 理解"检索-重排序"两阶段架构的必要性
2. 掌握 Cross-Encoder 重排序的原理
3. 对比 Bi-Encoder（检索）和 Cross-Encoder（重排序）的区别
4. 实现一个完整的 检索 → 重排序 管道

核心概念：
- Bi-Encoder：分别编码查询和文档，速度快但精度有限
- Cross-Encoder：同时编码查询+文档对，精度高但速度慢
- 两阶段架构：先用 Bi-Encoder 粗筛，再用 Cross-Encoder 精排

两阶段检索架构：
    用户查询
       │
       ▼
    ┌──────────────────┐
    │ 第一阶段：粗检索   │  Bi-Encoder / BM25
    │ 从 10000 篇中选 20 │  速度快，召回率高
    └────────┬─────────┘
             │ 20 个候选
             ▼
    ┌──────────────────┐
    │ 第二阶段：重排序   │  Cross-Encoder
    │ 从 20 篇中选 Top 3 │  精度高，速度慢
    └────────┬─────────┘
             │ 3 个最相关
             ▼
        送入 LLM 生成回答

运行方式：
    python 06_reranking.py
"""

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer

console = Console()


DOCUMENTS = [
    "RAG（Retrieval-Augmented Generation）通过检索外部知识来增强 LLM 的回答，解决了知识过时和幻觉问题。",
    "BM25 是经典的稀疏检索算法，基于词频-逆文档频率（TF-IDF）的改进版本，在关键词匹配场景表现优秀。",
    "向量数据库将文本编码为高维向量，通过余弦相似度进行语义检索，能捕获文本的深层语义关系。",
    "Cross-Encoder 重排序模型同时接收查询和文档作为输入，输出一个相关性分数，精度远高于 Bi-Encoder。",
    "文本分块的粒度直接影响检索质量：块太大导致噪声多，块太小导致上下文不足。需要根据场景调优。",
    "混合检索结合 BM25 和向量检索的优势，通过 RRF 等融合算法将两种检索结果合并排序。",
    "RAGAS 框架提供了 Faithfulness、Answer Relevancy、Context Precision 等指标来评估 RAG 系统。",
    "HyDE（Hypothetical Document Embeddings）先让 LLM 生成假设性答案，再用这个答案去检索，提升检索效果。",
    "Embedding 模型的选择对 RAG 性能影响巨大。中文场景推荐 BGE、M3E 等专门针对中文优化的模型。",
    "LLM 的上下文窗口限制了 RAG 能传入的文档量。需要通过重排序筛选最相关的文档，避免信息过载。",
]


# ============================================================
# 1. Bi-Encoder vs Cross-Encoder 原理对比
# ============================================================

def demo_bi_vs_cross_encoder():
    """
    Bi-Encoder：查询和文档分别编码，用向量距离衡量相关性
      - 优点：文档向量可以预计算，检索速度快
      - 缺点：查询和文档独立编码，无法捕获交互信息

    Cross-Encoder：查询和文档拼接后一起编码，直接输出相关性分数
      - 优点：能捕获查询-文档间的细粒度交互，精度高
      - 缺点：每个查询-文档对都要重新计算，速度慢
    """
    console.print(Panel("1️⃣  Bi-Encoder vs Cross-Encoder", style="bold cyan"))

    query = "如何提升 RAG 系统的检索质量？"

    # Bi-Encoder
    bi_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = bi_model.encode([query], normalize_embeddings=True)
    doc_embs = bi_model.encode(DOCUMENTS, normalize_embeddings=True)
    bi_scores = np.dot(doc_embs, query_emb.T).flatten()

    # Cross-Encoder
    cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc] for doc in DOCUMENTS]
    cross_scores = cross_model.predict(pairs)

    # 对比排名
    bi_ranking = np.argsort(bi_scores)[::-1]
    cross_ranking = np.argsort(cross_scores)[::-1]

    table = Table(title=f"查询：{query}")
    table.add_column("排名", width=4)
    table.add_column("Bi-Encoder", style="blue", max_width=40)
    table.add_column("Bi分数", style="blue", width=8)
    table.add_column("Cross-Encoder", style="green", max_width=40)
    table.add_column("Cross分数", style="green", width=8)

    for rank in range(5):
        bi_idx = bi_ranking[rank]
        cross_idx = cross_ranking[rank]
        table.add_row(
            str(rank + 1),
            DOCUMENTS[bi_idx][:38] + "...",
            f"{bi_scores[bi_idx]:.4f}",
            DOCUMENTS[cross_idx][:38] + "...",
            f"{cross_scores[cross_idx]:.4f}",
        )

    console.print(table)
    console.print("[dim]注意两种方法的排名差异 — Cross-Encoder 通常更准确[/dim]\n")


# ============================================================
# 2. 两阶段检索管道
# ============================================================

class TwoStageRetriever:
    """检索 → 重排序 两阶段管道"""

    def __init__(
        self,
        documents: list[str],
        bi_model_name: str = "all-MiniLM-L6-v2",
        cross_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.documents = documents
        self.bi_model = SentenceTransformer(bi_model_name)
        self.cross_model = CrossEncoder(cross_model_name)
        self.doc_embeddings = self.bi_model.encode(documents, normalize_embeddings=True)

    def search(
        self,
        query: str,
        first_stage_k: int = 10,
        final_k: int = 3,
    ) -> list[tuple[int, float]]:
        """
        两阶段检索：
        1. Bi-Encoder 粗检索 top-K 候选
        2. Cross-Encoder 对候选重排序
        """
        # 第一阶段：粗检索
        query_emb = self.bi_model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.doc_embeddings, query_emb.T).flatten()
        candidate_indices = np.argsort(similarities)[::-1][:first_stage_k]

        # 第二阶段：重排序
        pairs = [[query, self.documents[idx]] for idx in candidate_indices]
        rerank_scores = self.cross_model.predict(pairs)

        reranked = sorted(
            zip(candidate_indices, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(int(idx), float(score)) for idx, score in reranked[:final_k]]


# ============================================================
# 3. 对比实验
# ============================================================

def compare_with_without_reranking():
    """对比有无重排序的检索效果"""
    console.print(Panel("2️⃣  重排序效果对比", style="bold cyan"))

    retriever = TwoStageRetriever(DOCUMENTS)

    queries = [
        "重排序模型如何工作？",
        "RAG 系统的评估方法",
        "如何处理 LLM 上下文长度限制",
    ]

    for query in queries:
        console.print(f"\n[bold yellow]查询：{query}[/bold yellow]")

        # 无重排序
        query_emb = retriever.bi_model.encode([query], normalize_embeddings=True)
        sims = np.dot(retriever.doc_embeddings, query_emb.T).flatten()
        no_rerank = np.argsort(sims)[::-1][:3]

        # 有重排序
        reranked = retriever.search(query, first_stage_k=8, final_k=3)

        table = Table(show_header=True)
        table.add_column("排名", width=4)
        table.add_column("无重排序", style="red", max_width=50)
        table.add_column("有重排序", style="green", max_width=50)

        for rank in range(3):
            no_rr_doc = DOCUMENTS[no_rerank[rank]][:48] + "..."
            rr_idx, rr_score = reranked[rank]
            rr_doc = f"[{rr_score:.2f}] {DOCUMENTS[rr_idx][:42]}..."
            table.add_row(str(rank + 1), no_rr_doc, rr_doc)

        console.print(table)


# ============================================================
# 4. 演示
# ============================================================

if __name__ == "__main__":
    console.print(Panel("🏆 重排序：Cross-Encoder 精排", style="bold blue"))

    demo_bi_vs_cross_encoder()
    compare_with_without_reranking()

    console.print("\n[dim]下一步 → 07_query_transformation.py 学习查询改写技术[/dim]")
