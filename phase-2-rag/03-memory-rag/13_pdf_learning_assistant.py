"""
13_pdf_learning_assistant.py — 完整应用：PDF 智能学习助手

学习目标：
1. 将所有组件（记忆系统 + 生命周期 + 统一检索 + RAG）整合为完整应用
2. 实现 Markdown 结构感知分块，保留标题路径作为元数据
3. 构建 PDF 学习助手：加载文档 → 提问 → 记笔记 → 回顾 → 统计
4. 对比有/无记忆的 RAG 效果差异

完整应用架构：
    ┌─────────────────────────────────────────────────┐
    │              PDFLearningAssistant                │
    │                                                 │
    │  load_document()  ──▶  结构感知分块 + 索引       │
    │  ask()            ──▶  MemoryRAG.query()        │
    │  add_note()       ──▶  语义记忆存储              │
    │  review()         ──▶  记忆检索 + LLM 总结       │
    │  get_stats()      ──▶  学习统计                  │
    │                                                 │
    │  ┌─────────────────────────────────────────┐    │
    │  │            MemoryRAG (脚本 12)           │    │
    │  │  WorkingMemory + EpisodicMemory          │    │
    │  │  + SemanticMemory + UnifiedRetriever      │    │
    │  └─────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────┘

运行方式：
    python 13_pdf_learning_assistant.py
"""

import os
import re
import shutil
from datetime import datetime
from importlib import import_module

import chromadb
import litellm
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")

_m09 = import_module("09_memory_system")
_m11 = import_module("11_unified_retrieval")
_m12 = import_module("12_memory_enhanced_rag")

MemoryRAG = _m12.MemoryRAG
EmbeddingWithFallback = _m11.EmbeddingWithFallback


# ============================================================
# 1. Markdown 结构感知分块
# ============================================================

def chunk_markdown_with_paths(
    text: str, chunk_size: int = 500, overlap: int = 50,
) -> list[dict]:
    """Markdown 结构感知分块，保留标题路径

    与普通分块的区别：
    - 按标题层级（#/##/###）分割，保持语义完整性
    - 每个 chunk 携带 heading_path 元数据（如 "第1章 > 1.1 基础概念"）
    - 支持 overlap 保持上下文连续性

    返回: [{"content": str, "heading_path": str, "index": int}, ...]
    """
    lines = text.splitlines()
    heading_stack: list[str] = []
    paragraphs: list[dict] = []
    buf: list[str] = []

    def flush():
        if not buf:
            return
        content = "\n".join(buf).strip()
        if content:
            path = " > ".join(heading_stack) if heading_stack else ""
            paragraphs.append({"content": content, "heading_path": path})
        buf.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level - 1]
            heading_stack.append(title)
        elif stripped == "":
            flush()
        else:
            buf.append(line)
    flush()

    if not paragraphs:
        return [{"content": text, "heading_path": "", "index": 0}]

    # 按 chunk_size 合并段落
    chunks: list[dict] = []
    cur_parts: list[dict] = []
    cur_len = 0

    for para in paragraphs:
        p_len = len(para["content"])
        if cur_len + p_len > chunk_size and cur_parts:
            content = "\n\n".join(p["content"] for p in cur_parts)
            path = next(
                (p["heading_path"] for p in reversed(cur_parts)
                 if p["heading_path"]), "")
            chunks.append({
                "content": content,
                "heading_path": path,
                "index": len(chunks),
            })
            # overlap: 保留最后一个段落
            if overlap > 0 and cur_parts:
                last = cur_parts[-1]
                cur_parts = [last]
                cur_len = len(last["content"])
            else:
                cur_parts = []
                cur_len = 0

        cur_parts.append(para)
        cur_len += p_len

    if cur_parts:
        content = "\n\n".join(p["content"] for p in cur_parts)
        path = next(
            (p["heading_path"] for p in reversed(cur_parts)
             if p["heading_path"]), "")
        chunks.append({
            "content": content,
            "heading_path": path,
            "index": len(chunks),
        })

    return chunks


# ============================================================
# 2. PDFLearningAssistant
# ============================================================

class PDFLearningAssistant:
    """PDF 智能学习助手 — 整合 Memory + RAG 的完整应用"""

    def __init__(self, db_dir: str = "./memory_db"):
        self.db_dir = db_dir
        self._embedder = EmbeddingWithFallback()
        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(
            "learning_assistant", metadata={"hnsw:space": "cosine"})
        self._rag = MemoryRAG(self._collection, db_dir=db_dir)
        self._stats = {
            "start_time": datetime.now(),
            "documents_loaded": 0,
            "chunks_indexed": 0,
            "questions_asked": 0,
            "notes_added": 0,
        }
        self._current_doc: str | None = None

    def load_document(self, text: str, doc_name: str = "document") -> dict:
        """加载文档文本（已转为字符串），执行结构感知分块和索引"""
        chunks = chunk_markdown_with_paths(text, chunk_size=500)
        if not chunks:
            return {"success": False, "message": "文档为空"}

        contents = [c["content"] for c in chunks]
        embeddings = self._embedder.embed(contents)
        ids = [f"{doc_name}_chunk_{c['index']}" for c in chunks]
        metadatas = [{"heading_path": c["heading_path"],
                      "doc_name": doc_name,
                      "chunk_index": c["index"]}
                     for c in chunks]

        self._collection.add(
            ids=ids, embeddings=embeddings,
            documents=contents, metadatas=metadatas,
        )

        self._stats["documents_loaded"] += 1
        self._stats["chunks_indexed"] += len(chunks)
        self._current_doc = doc_name

        # 记录到情景记忆
        self._rag.episodic.add(
            f"加载了文档《{doc_name}》，共 {len(chunks)} 个分块",
            importance=0.9, session_id="learning",
        )

        return {
            "success": True,
            "message": f"已加载 {doc_name}，{len(chunks)} 个分块已索引",
            "chunks": len(chunks),
        }

    def ask(self, question: str) -> str:
        """向文档提问（记忆增强 RAG）"""
        self._stats["questions_asked"] += 1
        result = self._rag.query(question, use_mqe=False, use_hyde=False)
        return result["answer"]

    def add_note(self, content: str, concept: str = "general"):
        """添加学习笔记到语义记忆"""
        self._rag.semantic.add(
            content, importance=0.8,
            concept=concept, note_type="user_note",
        )
        self._stats["notes_added"] += 1

    def review(self, topic: str) -> str:
        """基于记忆生成学习回顾"""
        # 从各类记忆中检索相关内容
        episodic = self._rag.episodic.retrieve(topic, top_k=3)
        semantic = self._rag.semantic.retrieve(topic, top_k=3)

        parts = []
        for mem, score in episodic:
            parts.append(f"[经历] {mem.content}")
        for mem, score in semantic:
            parts.append(f"[知识] {mem.content}")

        if not parts:
            return f"暂无关于「{topic}」的学习记录"

        memory_text = "\n".join(parts)
        prompt = (f"请基于以下学习记录，生成关于「{topic}」的简要回顾：\n\n"
                  f"{memory_text}\n\n请用 3-5 句话总结关键要点。")
        try:
            resp = litellm.completion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return f"学习记录：\n{memory_text}"

    def get_stats(self) -> dict:
        duration = (datetime.now() - self._stats["start_time"]).total_seconds()
        memory_stats = self._rag.get_stats()
        return {
            "学习时长": f"{duration:.0f}秒",
            "加载文档": self._stats["documents_loaded"],
            "索引分块": self._stats["chunks_indexed"],
            "提问次数": self._stats["questions_asked"],
            "学习笔记": self._stats["notes_added"],
            "工作记忆": memory_stats["working_size"],
            "情景记忆": memory_stats["episodic_size"],
            "语义记忆": memory_stats["semantic_size"],
        }

    def clear(self):
        self._rag.clear()
        try:
            self._chroma.delete_collection("learning_assistant")
        except Exception:
            pass
        self._collection = self._chroma.get_or_create_collection(
            "learning_assistant", metadata={"hnsw:space": "cosine"})
        self._rag = MemoryRAG(self._collection, db_dir=self.db_dir)


# ============================================================
# 3. Demo
# ============================================================

SAMPLE_DOCUMENT = """# RAG 技术深度指南

## 1. RAG 概述

### 1.1 什么是 RAG

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索和文本生成的 AI 技术。
它的核心思想是：在大语言模型生成回答之前，先从外部知识库中检索相关信息，
然后将检索到的信息作为上下文提供给模型，从而生成更准确、更可靠的回答。

### 1.2 为什么需要 RAG

大语言模型面临两个根本性问题：
1. 知识过时：模型的知识停留在训练数据的截止日期
2. 幻觉问题：当模型不知道答案时，会自信地编造看起来合理的答案

RAG 通过引入外部知识源，有效解决了这两个问题。

## 2. RAG 核心流程

### 2.1 离线索引阶段

离线阶段的目标是将文档转换为可检索的向量索引：
- 文档加载：支持 PDF、Word、网页等多种格式
- 文本分块：将长文档切分为适合检索的小块
- 向量化：使用嵌入模型将文本转换为高维向量
- 存储：将向量存入向量数据库（如 ChromaDB）

### 2.2 在线查询阶段

在线阶段处理用户的实时查询：
- 查询向量化：将用户问题转换为向量
- 相似度检索：在向量数据库中找到最相关的文档块
- 上下文构建：将检索结果拼接为 LLM 的上下文
- 生成回答：LLM 基于上下文生成准确的回答

## 3. 高级检索策略

### 3.1 混合检索

混合检索结合稀疏检索（BM25）和稠密检索（向量），通过 RRF 融合算法取长补短。
BM25 擅长精确的关键词匹配，向量检索擅长语义理解，两者互补可显著提升检索质量。

### 3.2 查询改写

查询改写技术包括：
- HyDE：生成假设性答案文档，用答案去检索
- 多查询扩展（MQE）：将一个问题改写为多个变体
- Step-back：先问更抽象的问题获取背景知识

### 3.3 重排序

Cross-Encoder 重排序模型在初步检索后对候选文档进行精细化排序，
虽然计算成本较高，但能显著提升检索精度。

## 4. RAG 评估

### 4.1 RAGAS 框架

RAGAS 提供四个核心评估指标：
- 忠实度（Faithfulness）：答案是否基于检索到的文档
- 答案相关性（Answer Relevancy）：答案是否切题
- 上下文精确度（Context Precision）：检索结果是否精确
- 上下文召回率（Context Recall）：是否检索到了所有相关信息
"""


def demo_chunking():
    """演示 Markdown 结构感知分块"""
    console.print(Panel("[bold]Demo 1: Markdown 结构感知分块[/bold]"))

    chunks = chunk_markdown_with_paths(SAMPLE_DOCUMENT, chunk_size=300)
    table = Table(title=f"分块结果（共 {len(chunks)} 个）")
    table.add_column("#", width=3)
    table.add_column("标题路径", style="cyan", max_width=30)
    table.add_column("内容预览", max_width=40)
    table.add_column("长度", width=6)
    for c in chunks:
        table.add_row(
            str(c["index"]),
            c["heading_path"] or "(无)",
            c["content"][:40] + "...",
            str(len(c["content"])),
        )
    console.print(table)


def demo_study_session():
    """模拟完整学习会话"""
    console.print(Panel("[bold]Demo 2: 完整学习会话[/bold]"))

    assistant = PDFLearningAssistant(db_dir="./memory_db")
    assistant.clear()

    # Step 1: 加载文档
    console.print("  [bold]Step 1: 加载文档[/bold]")
    result = assistant.load_document(SAMPLE_DOCUMENT, doc_name="RAG指南")
    console.print(f"  {result['message']}")

    # Step 2: 提问
    console.print("\n  [bold]Step 2: 向文档提问[/bold]")
    questions = [
        "什么是 RAG？它解决了什么问题？",
        "RAG 的离线索引阶段包含哪些步骤？",
        "有哪些高级检索策略可以提升 RAG 效果？",
    ]
    for q in questions:
        console.print(f"\n  [cyan]Q: {q}[/cyan]")
        answer = assistant.ask(q)
        console.print(f"  A: {answer[:150]}...")

    # Step 3: 记笔记
    console.print("\n  [bold]Step 3: 添加学习笔记[/bold]")
    assistant.add_note(
        "RAG 的核心是「开卷考试」思路：先检索再生成",
        concept="RAG核心思想",
    )
    assistant.add_note(
        "混合检索 = BM25(关键词) + 向量(语义)，通过 RRF 融合",
        concept="混合检索",
    )
    console.print("  已添加 2 条学习笔记")

    # Step 4: 学习回顾
    console.print("\n  [bold]Step 4: 学习回顾[/bold]")
    review = assistant.review("RAG 检索策略")
    console.print(f"  {review[:200]}...")

    # Step 5: 统计
    console.print("\n  [bold]Step 5: 学习统计[/bold]")
    stats = assistant.get_stats()
    table = Table(title="学习统计")
    for k, v in stats.items():
        table.add_column(k, width=10)
    table.add_row(*[str(v) for v in stats.values()])
    console.print(table)

    assistant.clear()


def demo_compare_with_without_memory():
    """对比有/无记忆的 RAG 效果"""
    console.print(Panel("[bold]Demo 3: 有/无记忆的 RAG 效果对比[/bold]"))

    # 无记忆 RAG
    embedder = EmbeddingWithFallback()
    client = chromadb.Client()
    try:
        client.delete_collection("compare_plain")
    except Exception:
        pass
    col_plain = client.create_collection(
        "compare_plain", metadata={"hnsw:space": "cosine"})

    chunks = chunk_markdown_with_paths(SAMPLE_DOCUMENT, chunk_size=300)
    contents = [c["content"] for c in chunks]
    embeddings = embedder.embed(contents)
    col_plain.add(
        ids=[f"c_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=contents,
        metadatas=[{"index": i} for i in range(len(chunks))],
    )

    # 有记忆 RAG
    assistant = PDFLearningAssistant(db_dir="./memory_db")
    assistant.clear()
    assistant.load_document(SAMPLE_DOCUMENT, doc_name="RAG指南")

    # 先进行一些交互积累记忆
    assistant.ask("什么是 RAG？")
    assistant.add_note("RAG = 检索 + 增强 + 生成，核心是开卷考试思路")
    assistant.ask("有哪些检索策略？")

    # 对比问题
    question = "结合之前的讨论，如何构建高质量的 RAG 系统？"
    console.print(f"  [cyan]对比问题: {question}[/cyan]\n")

    # 无记忆：直接向量检索
    plain_retriever = _m11.UnifiedRetriever(col_plain)
    plain_results = plain_retriever.search(question, top_k=3,
                                           use_mqe=False, use_hyde=False)
    plain_context = "\n".join(r["content"][:80] for r in plain_results)
    console.print("  [bold]无记忆 RAG 检索到的上下文:[/bold]")
    console.print(f"  {plain_context[:200]}...")

    # 有记忆
    console.print("\n  [bold]有记忆 RAG 的回答:[/bold]")
    answer = assistant.ask(question)
    console.print(f"  {answer[:200]}...")

    stats = assistant.get_stats()
    console.print(f"\n  [dim]记忆状态: 工作={stats['工作记忆']}, "
                  f"情景={stats['情景记忆']}, 语义={stats['语义记忆']}[/dim]")
    assistant.clear()


def cleanup():
    if os.path.exists("./memory_db"):
        shutil.rmtree("./memory_db")
        console.print("\n[dim]已清理 memory_db 目录[/dim]")


if __name__ == "__main__":
    console.print(Panel(
        "[bold]13 — PDF 智能学习助手：Memory + RAG 完整应用[/bold]\n"
        "结构感知分块 | 记忆增强问答 | 学习笔记 | 知识回顾",
        style="blue",
    ))
    demo_chunking()
    demo_study_session()
    demo_compare_with_without_memory()
    cleanup()

