"""
03_embedding_vectorstore.py — Embedding 与向量数据库

学习目标：
1. 理解 Embedding 的本质：把文本变成数学向量，让"语义相似"变成"距离相近"
2. 掌握 sentence-transformers 本地 Embedding 模型的使用
3. 学会用 ChromaDB 存储和检索向量
4. 理解相似度搜索的原理（余弦相似度、欧氏距离）

核心概念：
- Embedding：文本 → 高维向量（如 384 维或 768 维）
- 语义相似度：向量空间中的距离 ≈ 语义上的相关性
- 向量数据库：专门为高维向量检索优化的数据库
- ChromaDB：轻量级向量数据库，适合开发和学习

向量检索原理：
    "什么是 RAG？"  ──Embedding──▶  [0.12, -0.34, 0.56, ...]
                                           │
                                    计算余弦相似度
                                           │
    ┌──────────────────────────────────────┼──────────────────┐
    │ 向量数据库                            ▼                  │
    │  [0.11, -0.33, 0.55, ...] "RAG 是检索增强生成..."  ← 最相似 │
    │  [0.78, 0.12, -0.45, ...] "Agent 是智能系统..."          │
    │  [-0.22, 0.67, 0.11, ...] "MCP 是工具调用标准..."        │
    └──────────────────────────────────────────────────────────┘

运行方式：
    python 03_embedding_vectorstore.py
"""

import os
from pathlib import Path

import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer

console = Console()


# ============================================================
# 1. 理解 Embedding
# ============================================================

def demo_embedding_basics():
    """
    Embedding 把文本映射到向量空间。
    语义相近的文本，向量也相近。
    """
    console.print(Panel("1️⃣  Embedding 基础", style="bold cyan"))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "什么是人工智能？",
        "AI 的定义是什么？",
        "今天天气怎么样？",
        "机器学习和深度学习的区别",
    ]

    embeddings = model.encode(sentences)

    console.print(f"模型: all-MiniLM-L6-v2")
    console.print(f"向量维度: {embeddings.shape[1]}")
    console.print(f"句子数量: {embeddings.shape[0]}\n")

    # 计算余弦相似度
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    table = Table(title="句子间余弦相似度")
    table.add_column("句子 A", style="cyan", max_width=25)
    table.add_column("句子 B", style="cyan", max_width=25)
    table.add_column("相似度", style="green")

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2)]
    for i, j in pairs:
        sim = cosine_similarity(embeddings[i], embeddings[j])
        table.add_row(sentences[i], sentences[j], f"{sim:.4f}")

    console.print(table)
    console.print("[dim]注意：语义相近的句子（前两句）相似度明显更高[/dim]\n")


# ============================================================
# 2. ChromaDB 向量数据库
# ============================================================

def demo_chromadb():
    """
    ChromaDB 是最简单的向量数据库之一。
    它内置了 Embedding 功能，可以直接存文本。
    """
    console.print(Panel("2️⃣  ChromaDB 向量数据库", style="bold cyan"))

    client = chromadb.Client()

    collection = client.create_collection(
        name="rag_demo",
        metadata={"hnsw:space": "cosine"},
    )

    documents = [
        "RAG（检索增强生成）是让 LLM 基于外部知识回答问题的技术。核心流程包括文档分块、向量化、检索和生成。",
        "ReAct 框架让 LLM 交替进行思考和行动。每次行动后观察结果，再决定下一步。这是最经典的 Agent 框架。",
        "向量数据库专门为高维向量的存储和检索而优化。常见的有 ChromaDB、Milvus、Pinecone、Weaviate 等。",
        "Embedding 模型将文本转换为高维向量。语义相近的文本在向量空间中距离更近。常用模型有 all-MiniLM-L6-v2。",
        "多 Agent 系统由多个专业化的 Agent 协作完成复杂任务。Manager Agent 负责任务分配，Worker Agent 负责执行。",
        "MCP（Model Context Protocol）是 Anthropic 提出的工具调用标准，让 Agent 可以统一调用各种外部工具和服务。",
        "文本分块是 RAG 管道的关键步骤。常见策略有固定大小分块、递归字符分块和语义分块。块的大小直接影响检索质量。",
        "LLM 的幻觉问题是指模型生成看似合理但实际错误的内容。RAG 通过提供真实文档作为上下文来缓解这个问题。",
    ]

    ids = [f"doc_{i}" for i in range(len(documents))]
    metadatas = [{"topic": "rag" if "RAG" in d or "检索" in d or "分块" in d or "幻觉" in d else "agent"} for d in documents]

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    console.print(f"已存入 {collection.count()} 个文档\n")

    # 基础查询
    console.print("[bold]基础语义搜索：[/bold]")
    results = collection.query(query_texts=["RAG 是怎么工作的？"], n_results=3)

    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
        console.print(f"  [{i+1}] 距离={dist:.4f} | {doc[:60]}...")

    # 带元数据过滤的查询
    console.print("\n[bold]带过滤的搜索（只搜 RAG 相关）：[/bold]")
    results = collection.query(
        query_texts=["文档处理技术"],
        n_results=3,
        where={"topic": "rag"},
    )

    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
        console.print(f"  [{i+1}] 距离={dist:.4f} | {doc[:60]}...")

    console.print()
    return collection


# ============================================================
# 3. 使用自定义 Embedding 模型
# ============================================================

def demo_custom_embedding():
    """
    ChromaDB 默认用 all-MiniLM-L6-v2，但你可以换成任何模型。
    中文场景推荐：BAAI/bge-small-zh-v1.5 或 shibing624/text2vec-base-chinese
    """
    console.print(Panel("3️⃣  自定义 Embedding 函数", style="bold cyan"))

    from chromadb.utils import embedding_functions

    # 使用 sentence-transformers 的模型
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.Client()
    collection = client.create_collection(
        name="custom_embedding_demo",
        embedding_function=ef,
    )

    collection.add(
        documents=["这是一个测试文档", "向量数据库很有用"],
        ids=["test_1", "test_2"],
    )

    results = collection.query(query_texts=["测试"], n_results=1)
    console.print(f"查询 '测试' → 最相关: {results['documents'][0][0]}")
    console.print(f"[dim]提示：中文场景建议使用专门的中文 Embedding 模型[/dim]\n")


# ============================================================
# 4. 持久化存储
# ============================================================

def demo_persistent_storage():
    """
    ChromaDB 支持持久化到磁盘，重启后数据不丢失。
    生产环境必须用持久化模式。
    """
    console.print(Panel("4️⃣  持久化存储", style="bold cyan"))

    db_path = "./chroma_db"
    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection("persistent_demo")
    collection.add(
        documents=["持久化测试文档"],
        ids=["persist_1"],
    )

    console.print(f"数据已持久化到: {db_path}/")
    console.print(f"集合中文档数: {collection.count()}")

    # 清理
    client.delete_collection("persistent_demo")
    import shutil
    shutil.rmtree(db_path, ignore_errors=True)
    console.print("[dim]（演示完毕，已清理持久化数据）[/dim]\n")


# ============================================================
# 5. 演示入口
# ============================================================

if __name__ == "__main__":
    console.print(Panel("🔢 Embedding 与向量数据库", style="bold blue"))

    demo_embedding_basics()
    demo_chromadb()
    demo_custom_embedding()
    demo_persistent_storage()

    console.print("[dim]下一步 → 04_naive_rag.py 把所有组件串起来，实现完整的 RAG[/dim]")
