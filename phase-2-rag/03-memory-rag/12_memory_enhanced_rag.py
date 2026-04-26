"""
12_memory_enhanced_rag.py — Memory + RAG 集成：记忆增强的检索生成

学习目标：
1. 理解记忆系统如何增强 RAG 的检索质量和个性化
2. 实现查询增强：用历史记忆上下文丰富用户查询
3. 实现知识积累：RAG 交互结果自动反馈到记忆系统
4. 构建完整的 MemoryRAG 类，串联所有组件
5. 观察多轮对话中记忆如何逐步改善回答质量

核心概念：
- 查询增强：检索前用记忆上下文补充查询信息
- 上下文融合：将检索文档和相关记忆合并为 LLM 上下文
- 知识积累：每次交互后将关键信息存入记忆，形成正反馈循环

Memory + RAG 集成架构：
    用户问题
       │
       ▼
    ┌──────────────────┐
    │  查询增强          │ ← 从情景/语义记忆中检索相关上下文
    │  (Memory Enrich)  │
    └────────┬─────────┘
             │ 增强后的查询
             ▼
    ┌──────────────────┐
    │  统一检索          │ ← MQE + HyDE + 候选池融合
    │  (Unified Search) │
    └────────┬─────────┘
             │ 检索文档 + 记忆上下文
             ▼
    ┌──────────────────┐
    │  LLM 生成回答     │
    └────────┬─────────┘
             │ 回答
             ▼
    ┌──────────────────┐
    │  存储交互          │ → 工作记忆 + 情景记忆 + 触发整合
    │  (Store & Learn)  │
    └──────────────────┘

运行方式：
    python 12_memory_enhanced_rag.py
"""

import os
import shutil
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

# 导入前序模块
_m09 = import_module("09_memory_system")
_m10 = import_module("10_memory_lifecycle")
_m11 = import_module("11_unified_retrieval")

MemoryItem = _m09.MemoryItem
WorkingMemory = _m09.WorkingMemory
EpisodicMemory = _m09.EpisodicMemory
SemanticMemory = _m09.SemanticMemory
MemoryLifecycleManager = _m10.MemoryLifecycleManager
MemoryConsolidator = _m10.MemoryConsolidator
UnifiedRetriever = _m11.UnifiedRetriever
EmbeddingWithFallback = _m11.EmbeddingWithFallback


# ============================================================
# 1. Prompt 模板
# ============================================================

MEMORY_RAG_PROMPT = """你是一个智能助手，请基于以下信息回答用户的问题。

## 检索到的文档
{retrieved_context}

## 相关记忆（历史交互中积累的知识）
{memory_context}

## 用户问题
{question}

请基于以上信息给出准确、有条理的回答。如果信息不足，请如实说明。"""


def call_llm(prompt: str) -> str:
    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM 调用失败: {e}]"


# ============================================================
# 2. MemoryRAG 核心类
# ============================================================

class MemoryRAG:
    """记忆增强的 RAG 系统

    将 WorkingMemory + EpisodicMemory + SemanticMemory + UnifiedRetriever
    整合为一个完整的问答系统，支持：
    - 查询增强：用记忆上下文丰富查询
    - 上下文融合：检索文档 + 记忆合并
    - 知识积累：交互结果自动存入记忆
    """

    def __init__(self, collection: chromadb.Collection,
                 db_dir: str = "./memory_db"):
        self.working = WorkingMemory(capacity=30, ttl_minutes=120)
        self.episodic = EpisodicMemory(
            db_path=f"{db_dir}/rag_episodic.db",
            collection_name="rag_episodic",
        )
        self.semantic = SemanticMemory(
            db_path=f"{db_dir}/rag_semantic.db",
            collection_name="rag_semantic",
        )
        self.retriever = UnifiedRetriever(collection)
        self.consolidator = MemoryConsolidator(importance_threshold=0.7)
        self._session_id = "default"
        self._turn_count = 0

    def set_session(self, session_id: str):
        self._session_id = session_id
        self._turn_count = 0

    def query(self, question: str, use_mqe: bool = True,
              use_hyde: bool = False, top_k: int = 5) -> dict:
        """完整的记忆增强 RAG 查询流程"""
        self._turn_count += 1

        # Step 1: 用记忆增强查询
        enriched_query = self._enrich_query(question)

        # Step 2: 统一检索
        results = self.retriever.search(
            enriched_query, top_k=top_k,
            use_mqe=use_mqe, use_hyde=use_hyde,
        )

        # Step 3: 构建上下文（检索文档 + 记忆）
        retrieved_ctx = self._format_retrieved(results)
        memory_ctx = self._get_memory_context(question)

        # Step 4: LLM 生成回答
        prompt = MEMORY_RAG_PROMPT.format(
            retrieved_context=retrieved_ctx,
            memory_context=memory_ctx,
            question=question,
        )
        answer = call_llm(prompt)

        # Step 5: 存储交互到记忆
        self._store_interaction(question, answer, results)

        return {
            "answer": answer,
            "sources": [r["content"][:60] for r in results[:3]],
            "memory_used": bool(memory_ctx.strip()),
            "turn": self._turn_count,
        }

    def _enrich_query(self, question: str) -> str:
        """用记忆上下文增强原始查询"""
        # 从工作记忆获取最近对话上下文
        recent = self.working.retrieve(question, top_k=2)
        # 从语义记忆获取相关知识
        knowledge = self.semantic.retrieve(question, top_k=2)

        context_parts = []
        for mem, _ in recent:
            context_parts.append(mem.content)
        for mem, _ in knowledge:
            context_parts.append(mem.content)

        if not context_parts:
            return question
        context = "；".join(context_parts)
        return f"{question}（背景：{context}）"

    def _get_memory_context(self, question: str) -> str:
        """获取与问题相关的记忆上下文"""
        parts = []
        episodic_results = self.episodic.retrieve(question, top_k=2)
        for mem, score in episodic_results:
            if score > 0.3:
                parts.append(f"[历史] {mem.content}")

        semantic_results = self.semantic.retrieve(question, top_k=2)
        for mem, score in semantic_results:
            if score > 0.3:
                parts.append(f"[知识] {mem.content}")

        return "\n".join(parts) if parts else "（暂无相关记忆）"

    def _store_interaction(self, question: str, answer: str,
                           results: list[dict]):
        """将交互存入记忆系统"""
        # 工作记忆：记录当前对话
        self.working.add(
            f"Q: {question}\nA: {answer[:100]}",
            importance=0.6, session_id=self._session_id,
        )
        # 情景记忆：记录交互事件
        self.episodic.add(
            f"用户问了「{question}」，回答涉及 {len(results)} 个文档",
            importance=0.7, session_id=self._session_id,
        )
        # 每 3 轮触发一次整合
        if self._turn_count % 3 == 0:
            self.consolidator.consolidate_working_to_episodic(
                self.working, self.episodic, self._session_id)

    @staticmethod
    def _format_retrieved(results: list[dict]) -> str:
        if not results:
            return "（未检索到相关文档）"
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[文档{i}] (相关度: {r['score']:.2f}) {r['content']}")
        return "\n".join(parts)

    def add_knowledge(self, content: str, importance: float = 0.8):
        """手动添加知识到语义记忆"""
        self.semantic.add(content, importance=importance)

    def get_stats(self) -> dict:
        return {
            "working_size": self.working.size(),
            "episodic_size": self.episodic.size(),
            "semantic_size": self.semantic.size(),
            "session": self._session_id,
            "turns": self._turn_count,
        }

    def clear(self):
        self.working.clear()
        self.episodic.clear()
        self.semantic.clear()


# ============================================================
# 3. Demo
# ============================================================

SAMPLE_DOCS = [
    "RAG（检索增强生成）通过检索外部知识来增强大语言模型的回答准确性，解决知识过时和幻觉问题。",
    "向量数据库（如 ChromaDB、Qdrant）将文本转换为高维向量存储，支持高效的语义相似度搜索。",
    "文本分块是 RAG 的关键步骤，常见策略包括固定大小分块、递归字符分块和 Markdown 感知分块。",
    "BM25 是基于词频统计的稀疏检索算法，与向量检索互补，混合使用可提升召回率。",
    "HyDE 通过生成假设性答案文档来检索，缩小查询与文档之间的语义鸿沟。",
    "RAGAS 评估框架提供忠实度、答案相关性、上下文精确度和上下文召回率四个核心指标。",
    "Cross-Encoder 重排序在初步检索后对候选文档精细排序，显著提升检索精度。",
    "记忆系统让 Agent 能够记住历史交互，包括工作记忆（短期）、情景记忆（事件）和语义记忆（知识）。",
    "记忆整合将重要的短期记忆提升为长期记忆，遗忘机制清理低价值信息保持系统高效。",
    "多查询扩展（MQE）将一个问题改写为多个变体查询，从不同角度覆盖更多相关文档。",
]


def _build_collection() -> chromadb.Collection:
    embedder = EmbeddingWithFallback()
    client = chromadb.Client()
    try:
        client.delete_collection("memory_rag_demo")
    except Exception:
        pass
    col = client.create_collection("memory_rag_demo",
                                   metadata={"hnsw:space": "cosine"})
    embeddings = embedder.embed(SAMPLE_DOCS)
    col.add(
        ids=[f"doc_{i}" for i in range(len(SAMPLE_DOCS))],
        embeddings=embeddings,
        documents=SAMPLE_DOCS,
        metadatas=[{"index": i} for i in range(len(SAMPLE_DOCS))],
    )
    return col


def demo_multi_turn():
    """多轮对话演示：观察记忆如何逐步积累"""
    console.print(Panel("[bold]Demo 1: 多轮对话 — 记忆逐步积累[/bold]"))

    col = _build_collection()
    rag = MemoryRAG(col, db_dir="./memory_db")
    rag.clear()
    rag.set_session("demo_session_1")

    questions = [
        "什么是 RAG？它解决了什么问题？",
        "RAG 系统中文本分块有哪些策略？",
        "如何评估 RAG 系统的质量？",
        "结合前面的讨论，如何构建一个高质量的 RAG 系统？",
    ]

    for q in questions:
        console.print(f"\n  [bold cyan]Q: {q}[/bold cyan]")
        result = rag.query(q, use_mqe=False, use_hyde=False)
        console.print(f"  A: {result['answer'][:150]}...")
        console.print(f"  [dim]Turn {result['turn']} | "
                      f"记忆参与: {result['memory_used']} | "
                      f"来源: {len(result['sources'])} 篇[/dim]")

    stats = rag.get_stats()
    table = Table(title="会话结束时的记忆状态")
    for k, v in stats.items():
        table.add_column(k, width=12)
    table.add_row(*[str(v) for v in stats.values()])
    console.print(table)
    rag.clear()


def demo_knowledge_accumulation():
    """知识积累演示：手动添加知识后观察检索改善"""
    console.print(Panel("[bold]Demo 2: 知识积累 — 记忆改善检索[/bold]"))

    col = _build_collection()
    rag = MemoryRAG(col, db_dir="./memory_db")
    rag.clear()
    rag.set_session("demo_session_2")

    question = "如何优化 RAG 的检索效果？"

    # 第一次查询（无记忆）
    console.print("  [bold]第一次查询（无记忆积累）:[/bold]")
    r1 = rag.query(question, use_mqe=False, use_hyde=False)
    console.print(f"  A: {r1['answer'][:120]}...")

    # 添加领域知识
    rag.add_knowledge("优化 RAG 检索的关键方法：1) 混合检索(BM25+向量) "
                      "2) 查询改写(HyDE/MQE) 3) 重排序(Cross-Encoder) "
                      "4) 优化分块策略")
    rag.add_knowledge("RAG 评估应关注：忠实度（答案是否基于文档）、"
                      "相关性（答案是否切题）、上下文质量（检索是否准确）")

    # 第二次查询（有记忆）
    console.print("\n  [bold]第二次查询（添加知识后）:[/bold]")
    r2 = rag.query(question, use_mqe=False, use_hyde=False)
    console.print(f"  A: {r2['answer'][:120]}...")
    console.print(f"  [dim]记忆参与: {r2['memory_used']}[/dim]")
    rag.clear()


def cleanup():
    if os.path.exists("./memory_db"):
        shutil.rmtree("./memory_db")
        console.print("\n[dim]已清理 memory_db 目录[/dim]")


if __name__ == "__main__":
    console.print(Panel(
        "[bold]12 — Memory + RAG 集成：记忆增强的检索生成[/bold]\n"
        "查询增强 | 上下文融合 | 知识积累 | 多轮对话",
        style="blue",
    ))
    demo_multi_turn()
    demo_knowledge_accumulation()
    cleanup()

