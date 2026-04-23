"""
04_naive_rag.py — 最简 RAG：检索 + 生成

学习目标：
1. 把前三节的组件串起来，实现一个完整的 RAG 管道
2. 理解 RAG 的核心流程：加载 → 分块 → 向量化 → 检索 → 生成
3. 观察 RAG 如何让 LLM 基于文档回答问题（而不是靠幻觉）
4. 理解 Naive RAG 的局限性，为高级 RAG 做铺垫

完整 RAG 管道：
    ┌──────────────────────────────────────────────────────────────┐
    │                    离线索引阶段（Indexing）                     │
    │                                                              │
    │  文档 ──▶ 加载 ──▶ 分块 ──▶ Embedding ──▶ 存入向量数据库       │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │                    在线查询阶段（Querying）                     │
    │                                                              │
    │  用户问题 ──▶ Embedding ──▶ 向量检索 ──▶ 取回相关文档           │
    │                                              │               │
    │                                              ▼               │
    │                                    拼接 Prompt + 文档         │
    │                                              │               │
    │                                              ▼               │
    │                                         LLM 生成回答          │
    └──────────────────────────────────────────────────────────────┘

运行方式：
    python 04_naive_rag.py
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()

console = Console()


# ============================================================
# 1. 索引阶段：加载 → 分块 → 存入向量库
# ============================================================

def build_index(collection_name: str = "naive_rag") -> chromadb.Collection:
    """构建向量索引"""
    console.print("[bold]📦 索引阶段[/bold]")

    # 加载文档
    sample_path = Path("sample_data/ai_agent_overview.txt")
    if not sample_path.exists():
        from importlib import import_module
        mod = import_module("01_document_loading")
        mod.create_sample_documents()

    text = sample_path.read_text(encoding="utf-8")
    console.print(f"  ✓ 加载文档: {len(text)} 字符")

    # 分块
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    console.print(f"  ✓ 分块完成: {len(chunks)} 个块")

    # 存入向量库
    client = chromadb.Client()

    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"chunk_index": i, "source": "ai_agent_overview.txt"} for i in range(len(chunks))],
    )
    console.print(f"  ✓ 向量化并存入 ChromaDB: {collection.count()} 条记录\n")

    return collection


# ============================================================
# 2. 检索阶段：找到最相关的文档块
# ============================================================

def retrieve(collection: chromadb.Collection, query: str, top_k: int = 3) -> list[str]:
    """语义检索：找到与问题最相关的文档块"""
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]


# ============================================================
# 3. 生成阶段：用 LLM 基于检索到的文档生成回答
# ============================================================

RAG_PROMPT_TEMPLATE = """你是一个专业的 AI 技术助手。请基于以下参考文档回答用户的问题。

要求：
1. 只基于提供的参考文档回答，不要编造信息
2. 如果文档中没有相关信息，明确说"根据提供的文档，我无法回答这个问题"
3. 回答要简洁准确，可以引用文档中的关键内容

参考文档：
{context}

用户问题：{question}

回答："""


def generate_answer(query: str, context_docs: list[str]) -> str:
    """调用 LLM 生成回答"""
    import litellm

    context = "\n\n---\n\n".join(
        f"[文档 {i+1}]\n{doc}" for i, doc in enumerate(context_docs)
    )

    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

    response = litellm.completion(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content


# ============================================================
# 4. 完整 RAG 管道
# ============================================================

def rag_query(collection: chromadb.Collection, question: str) -> str:
    """完整的 RAG 查询流程"""
    console.print(f"\n[bold yellow]❓ 问题：{question}[/bold yellow]")

    # 检索
    docs = retrieve(collection, question, top_k=3)
    console.print(f"[dim]  检索到 {len(docs)} 个相关文档块：[/dim]")
    for i, doc in enumerate(docs):
        preview = doc[:80].replace("\n", " ")
        console.print(f"[dim]    [{i+1}] {preview}...[/dim]")

    # 生成
    answer = generate_answer(question, docs)

    console.print(Panel(Markdown(answer), title="💡 回答", border_style="green"))
    return answer


# ============================================================
# 5. 对比实验：有 RAG vs 无 RAG
# ============================================================

def compare_with_without_rag(collection: chromadb.Collection, question: str):
    """对比有无 RAG 的回答质量"""
    import litellm

    console.print(Panel(f"对比实验：{question}", style="bold magenta"))

    # 无 RAG：直接问 LLM
    console.print("[bold red]❌ 无 RAG（直接问 LLM）：[/bold red]")
    response = litellm.completion(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": question}],
        temperature=0.3,
    )
    console.print(f"  {response.choices[0].message.content[:200]}...\n")

    # 有 RAG
    console.print("[bold green]✅ 有 RAG（基于文档回答）：[/bold green]")
    rag_query(collection, question)


# ============================================================
# 6. Naive RAG 的局限性
# ============================================================

def show_limitations():
    """展示 Naive RAG 的常见问题"""
    console.print(Panel("⚠️  Naive RAG 的局限性", style="bold red"))

    limitations = [
        ("检索质量", "语义搜索可能遗漏关键文档，尤其是查询和文档用词不同时"),
        ("上下文窗口", "检索到的文档可能超出 LLM 上下文限制"),
        ("信息整合", "答案可能需要综合多个文档，但 Naive RAG 只是简单拼接"),
        ("查询理解", "用户的问题可能模糊或多义，直接检索效果差"),
        ("排序质量", "向量相似度不等于相关性，需要重排序"),
    ]

    table = console.status("")
    for problem, desc in limitations:
        console.print(f"  • [bold]{problem}[/bold]: {desc}")

    console.print("\n[dim]这些问题将在 02-advanced-rag 中逐一解决[/dim]")


# ============================================================
# 7. 演示入口
# ============================================================

if __name__ == "__main__":
    console.print(Panel("🔗 Naive RAG：完整的检索增强生成管道", style="bold blue"))

    # 构建索引
    collection = build_index()

    # RAG 问答
    questions = [
        "什么是 RAG？它解决了什么问题？",
        "ReAct 框架的工作原理是什么？",
        "多 Agent 系统是如何协作的？",
    ]

    for q in questions:
        rag_query(collection, q)

    # 展示局限性
    show_limitations()

    console.print("\n[dim]Phase 2 基础部分完成！下一步 → 02-advanced-rag/ 学习高级 RAG 技术[/dim]")
