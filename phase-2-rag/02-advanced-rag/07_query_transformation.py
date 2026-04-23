"""
07_query_transformation.py — 查询改写：HyDE、多查询、Step-back

学习目标：
1. 理解用户查询和文档之间的"语义鸿沟"问题
2. 掌握三种查询改写技术：HyDE、多查询扩展、Step-back 提问
3. 学会用 LLM 来优化检索查询
4. 理解查询改写在 RAG 管道中的位置

核心概念：
- 语义鸿沟：用户的提问方式和文档的表述方式往往不同
- HyDE：让 LLM 先生成假设性答案，用答案去检索（答案和文档更像）
- 多查询：把一个问题改写成多个不同角度的查询，扩大检索覆盖面
- Step-back：先问一个更抽象的问题，获取背景知识，再回答具体问题

查询改写在 RAG 管道中的位置：
    用户问题
       │
       ▼
    ┌──────────────────┐
    │   查询改写        │  ← 你在这里
    │  Query Transform │
    └────────┬─────────┘
             │ 改写后的查询（可能多个）
             ▼
    ┌──────────────────┐
    │   检索 + 重排序   │
    └────────┬─────────┘
             │
             ▼
        LLM 生成回答

运行方式：
    python 07_query_transformation.py
"""

import os

import chromadb
import litellm
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer

load_dotenv()

console = Console()

LLM_MODEL = "deepseek/deepseek-chat"


# ============================================================
# 准备向量检索器
# ============================================================

DOCUMENTS = [
    "RAG 通过检索外部知识来增强 LLM 的回答，核心流程是：分块 → 向量化 → 检索 → 生成。",
    "BM25 基于词频和逆文档频率计算相关性，是经典的稀疏检索算法。",
    "Cross-Encoder 重排序模型同时编码查询和文档，输出精确的相关性分数。",
    "文本分块策略包括固定大小、递归字符、语义分块等，块的大小直接影响检索质量。",
    "混合检索结合 BM25 和向量检索，通过 RRF 融合排序，覆盖精确匹配和语义匹配。",
    "RAGAS 提供 Faithfulness、Answer Relevancy 等指标评估 RAG 系统质量。",
    "HyDE 先让 LLM 生成假设性答案文档，再用这个假设文档去检索真实文档。",
    "多查询扩展将用户问题改写为多个不同角度的查询，合并检索结果以提高召回率。",
    "Step-back prompting 先问一个更抽象的问题获取背景知识，再回答具体问题。",
    "向量数据库使用 ANN 算法（如 HNSW）实现高效的近似最近邻搜索。",
]


def build_retriever() -> tuple[SentenceTransformer, any]:
    """构建简单的向量检索器"""
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = model.encode(DOCUMENTS, normalize_embeddings=True)
    return model, doc_embeddings


def vector_search(model, doc_embeddings, query: str, top_k: int = 3) -> list[tuple[int, float]]:
    import numpy as np

    query_emb = model.encode([query], normalize_embeddings=True)
    sims = np.dot(doc_embeddings, query_emb.T).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(int(idx), float(sims[idx])) for idx in top_indices]


def call_llm(prompt: str) -> str:
    response = litellm.completion(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# 1. HyDE（Hypothetical Document Embeddings）
# ============================================================

def hyde_transform(query: str) -> str:
    """
    HyDE 的核心思想：
    用户问题 → LLM 生成假设性答案 → 用答案去检索

    为什么有效？因为 LLM 生成的"答案"在表述风格上更接近文档，
    比用户的"问题"更容易匹配到相关文档。
    """
    prompt = f"""请针对以下问题，写一段简短的假设性答案（约 100 字）。
不需要完全准确，只需要包含可能相关的关键概念和术语。

问题：{query}

假设性答案："""

    return call_llm(prompt)


def demo_hyde(model, doc_embeddings):
    console.print(Panel("1️⃣  HyDE（假设性文档嵌入）", style="bold cyan"))

    query = "怎么让检索结果更准确？"
    console.print(f"原始查询：{query}")

    hypothetical_doc = hyde_transform(query)
    console.print(f"HyDE 生成的假设文档：{hypothetical_doc[:150]}...\n")

    # 对比
    original_results = vector_search(model, doc_embeddings, query, top_k=3)
    hyde_results = vector_search(model, doc_embeddings, hypothetical_doc, top_k=3)

    table = Table(title="HyDE 效果对比")
    table.add_column("排名", width=4)
    table.add_column("原始查询结果", style="red", max_width=45)
    table.add_column("HyDE 查询结果", style="green", max_width=45)

    for rank in range(3):
        orig_idx, orig_score = original_results[rank]
        hyde_idx, hyde_score = hyde_results[rank]
        table.add_row(
            str(rank + 1),
            f"[{orig_score:.3f}] {DOCUMENTS[orig_idx][:38]}...",
            f"[{hyde_score:.3f}] {DOCUMENTS[hyde_idx][:38]}...",
        )

    console.print(table)
    console.print()


# ============================================================
# 2. 多查询扩展（Multi-Query）
# ============================================================

def multi_query_transform(query: str, n: int = 3) -> list[str]:
    """
    将一个查询改写为多个不同角度的查询。
    每个查询从不同角度描述同一个信息需求，扩大检索覆盖面。
    """
    prompt = f"""请将以下用户问题改写为 {n} 个不同角度的搜索查询。
每个查询应该从不同的角度描述同一个信息需求。
每行一个查询，不要编号。

用户问题：{query}

改写后的查询："""

    result = call_llm(prompt)
    queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
    return queries[:n]


def demo_multi_query(model, doc_embeddings):
    console.print(Panel("2️⃣  多查询扩展", style="bold cyan"))

    query = "RAG 系统有哪些常见问题？"
    console.print(f"原始查询：{query}\n")

    expanded_queries = multi_query_transform(query)
    console.print("扩展后的查询：")
    for i, q in enumerate(expanded_queries):
        console.print(f"  {i+1}. {q}")

    # 合并多个查询的检索结果
    all_results: dict[int, float] = {}
    for eq in expanded_queries:
        results = vector_search(model, doc_embeddings, eq, top_k=3)
        for idx, score in results:
            if idx not in all_results or score > all_results[idx]:
                all_results[idx] = score

    merged = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:5]

    console.print(f"\n合并后的检索结果（去重）：")
    for rank, (idx, score) in enumerate(merged):
        console.print(f"  [{rank+1}] ({score:.3f}) {DOCUMENTS[idx][:60]}...")

    console.print()


# ============================================================
# 3. Step-back Prompting
# ============================================================

def stepback_transform(query: str) -> str:
    """
    Step-back：先问一个更抽象/更基础的问题。
    例如："Python 的 GIL 如何影响多线程性能？"
    → Step-back："Python 的并发模型是什么？"

    获取背景知识后，再回答具体问题。
    """
    prompt = f"""请将以下具体问题转化为一个更抽象、更基础的问题。
这个抽象问题应该能帮助获取回答原始问题所需的背景知识。

具体问题：{query}

抽象问题（一句话）："""

    return call_llm(prompt)


def demo_stepback(model, doc_embeddings):
    console.print(Panel("3️⃣  Step-back Prompting", style="bold cyan"))

    query = "Cross-Encoder 重排序比 Bi-Encoder 慢多少？"
    console.print(f"原始查询：{query}")

    stepback_query = stepback_transform(query)
    console.print(f"Step-back 查询：{stepback_query}\n")

    original_results = vector_search(model, doc_embeddings, query, top_k=3)
    stepback_results = vector_search(model, doc_embeddings, stepback_query, top_k=3)

    table = Table(title="Step-back 效果对比")
    table.add_column("排名", width=4)
    table.add_column("原始查询结果", style="red", max_width=45)
    table.add_column("Step-back 结果", style="green", max_width=45)

    for rank in range(3):
        orig_idx, orig_score = original_results[rank]
        sb_idx, sb_score = stepback_results[rank]
        table.add_row(
            str(rank + 1),
            f"[{orig_score:.3f}] {DOCUMENTS[orig_idx][:38]}...",
            f"[{sb_score:.3f}] {DOCUMENTS[sb_idx][:38]}...",
        )

    console.print(table)


# ============================================================
# 4. 演示入口
# ============================================================

if __name__ == "__main__":
    console.print(Panel("🔄 查询改写技术", style="bold blue"))

    console.print("初始化向量检索器...")
    model, doc_embeddings = build_retriever()
    console.print()

    demo_hyde(model, doc_embeddings)
    demo_multi_query(model, doc_embeddings)
    demo_stepback(model, doc_embeddings)

    console.print("\n[dim]下一步 → 08_ragas_evaluation.py 学习 RAG 系统评估[/dim]")
