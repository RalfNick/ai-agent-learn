"""
04_agentic_rag.py — LangGraph Agentic RAG：从线性管道到自适应图

设计思考：
Phase 2 的 RAG 是线性管道：查询 → 检索 → 生成 → 返回
这在简单场景下够用，但面对复杂查询时有三个问题：
1. 检索质量不确定 —— 检索到的文档可能不相关，但管道不会重试
2. 查询可能需要改写 —— 用户的原始问题不一定适合直接检索
3. 生成可能有幻觉 —— 没有自我检查机制

Agentic RAG 用 LangGraph 把线性管道变成自适应图：
- 查询分析节点：判断是否需要检索，还是可以直接回答
- 检索节点：执行向量搜索
- 评分节点：评估检索结果质量，不合格则改写查询重试
- 生成节点：基于检索结果生成回答
- 幻觉检查节点：验证回答是否有依据

这就是 Phase 2 → Phase 3 的桥梁：
  线性 RAG（Phase 2）→ 自适应 RAG（Phase 3）→ 生产级 RAG（Phase 6）

运行方式：
    python 04_agentic_rag.py
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()


# ── 1. 模拟知识库 ──────────────────────────────────────────────
# 实际项目中这里是向量数据库（Chroma/Milvus），这里用字典模拟
# 重点是展示 LangGraph 的图结构，不是 RAG 的检索细节（Phase 2 已覆盖）

KNOWLEDGE_BASE = {
    "langgraph": "LangGraph 是 LangChain 团队开发的 Agent 编排框架，核心概念是状态图（StateGraph）。"
                 "它用有向图表示 Agent 工作流，节点是处理函数，边是控制流。"
                 "支持条件路由、循环、检查点持久化和人机协作。"
                 "2026 年 PyPI 月下载量超过 4700 万，是企业级 Agent 开发的主流选择。",
    "crewai": "CrewAI 是一个多智能体协作框架，核心概念是 Agent（角色）、Task（任务）、Crew（团队）。"
              "它用组织隐喻来建模多 Agent 协作：每个 Agent 有角色、目标和背景故事。"
              "支持顺序执行和层级委派两种流程模式。适合快速原型化多角色工作流。",
    "mcp": "MCP（Model Context Protocol）是 Anthropic 提出的 AI 工具集成协议。"
           "定义了 Tools、Resources、Prompts 三大原语。"
           "2025 年 11 月捐赠给 Linux Foundation，OpenAI/Anthropic/Block 联合推动。"
           "已有 1000+ MCP Server，支持 Claude/Cursor/VS Code 等工具。",
    "rag": "RAG（Retrieval-Augmented Generation）是检索增强生成技术。"
           "核心流程：文档分块 → 向量化 → 存入向量数据库 → 检索 → 生成。"
           "优化方向包括混合检索（向量+BM25）、Rerank 重排序、HyDE 假设文档嵌入。"
           "评估指标：Faithfulness、Answer Relevancy、Context Precision、Context Recall。",
    "agent_security": "Agent 安全防护包括六层架构：输入验证 → Prompt 隔离 → 工具授权 → "
                      "输出过滤 → 执行沙箱 → 运行时监控。"
                      "主要威胁是 Prompt 注入（直接注入和间接注入）。"
                      "防御手段包括 Guardrails、输入清洗、权限最小化。",
}


def simple_retrieve(query: str, top_k: int = 2) -> list[dict]:
    """简单的关键词匹配检索（模拟向量检索）"""
    results = []
    query_lower = query.lower()
    for topic, content in KNOWLEDGE_BASE.items():
        score = sum(1 for word in query_lower.split() if word in topic or word in content.lower())
        if score > 0:
            results.append({"topic": topic, "content": content, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ── 2. 状态定义 ─────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str              # 用户原始问题
    rewritten_query: str       # 改写后的检索查询
    documents: list[dict]      # 检索到的文档
    doc_relevance: str         # 文档相关性评估（relevant/irrelevant）
    answer: str                # 生成的回答
    hallucination_check: str   # 幻觉检查结果（grounded/hallucinated）
    retry_count: int           # 重试次数


# ── 3. LLM ──────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.1,
)


# ── 4. 节点函数 ─────────────────────────────────────────────────

def query_analyzer(state: RAGState) -> dict:
    """查询分析：改写用户问题为更适合检索的形式"""
    response = llm.invoke([
        SystemMessage(content=(
            "你是查询改写专家。将用户问题改写为更适合知识库检索的关键词查询。"
            "只输出改写后的查询，不要解释。"
        )),
        HumanMessage(content=f"用户问题: {state['question']}"),
    ])
    rewritten = response.content.strip()
    console.print(f"  [cyan]查询改写:[/cyan] {state['question']} → {rewritten}")
    return {"rewritten_query": rewritten}


def retriever(state: RAGState) -> dict:
    """检索节点：从知识库检索相关文档"""
    query = state.get("rewritten_query") or state["question"]
    docs = simple_retrieve(query)
    console.print(f"  [cyan]检索到 {len(docs)} 篇文档:[/cyan] {[d['topic'] for d in docs]}")
    return {"documents": docs}


def doc_grader(state: RAGState) -> dict:
    """文档评分：评估检索结果是否与问题相关"""
    docs = state["documents"]
    if not docs:
        return {"doc_relevance": "irrelevant"}

    doc_text = "\n".join(d["content"] for d in docs)
    response = llm.invoke([
        SystemMessage(content=(
            "评估检索到的文档是否与用户问题相关。"
            "只回答 'relevant' 或 'irrelevant'。"
        )),
        HumanMessage(content=f"问题: {state['question']}\n\n文档:\n{doc_text}"),
    ])
    relevance = "relevant" if "relevant" in response.content.lower() else "irrelevant"
    console.print(f"  [cyan]文档相关性:[/cyan] {relevance}")
    return {"doc_relevance": relevance}


def generator(state: RAGState) -> dict:
    """生成节点：基于检索文档生成回答"""
    doc_text = "\n\n".join(d["content"] for d in state["documents"])
    response = llm.invoke([
        SystemMessage(content="基于提供的参考文档回答问题。如果文档信息不足，明确说明。"),
        HumanMessage(content=f"问题: {state['question']}\n\n参考文档:\n{doc_text}"),
    ])
    return {"answer": response.content}


def hallucination_checker(state: RAGState) -> dict:
    """幻觉检查：验证回答是否有文档依据"""
    doc_text = "\n".join(d["content"] for d in state["documents"])
    response = llm.invoke([
        SystemMessage(content=(
            "检查回答是否完全基于提供的文档。"
            "如果回答中的所有事实都能在文档中找到依据，回答 'grounded'。"
            "如果回答包含文档中没有的信息，回答 'hallucinated'。"
        )),
        HumanMessage(content=(
            f"文档:\n{doc_text}\n\n回答:\n{state['answer']}"
        )),
    ])
    check = "grounded" if "grounded" in response.content.lower() else "hallucinated"
    console.print(f"  [cyan]幻觉检查:[/cyan] {check}")
    return {"hallucination_check": check}


def query_rewriter(state: RAGState) -> dict:
    """查询重写：检索结果不佳时，换个角度重新查询"""
    response = llm.invoke([
        SystemMessage(content="用户的问题检索效果不好，请用不同的关键词重写查询。只输出新查询。"),
        HumanMessage(content=f"原始问题: {state['question']}\n上次查询: {state.get('rewritten_query', '')}"),
    ])
    retry = state.get("retry_count", 0) + 1
    console.print(f"  [yellow]重写查询 (第{retry}次):[/yellow] {response.content.strip()}")
    return {"rewritten_query": response.content.strip(), "retry_count": retry}


# ── 5. 路由函数 ─────────────────────────────────────────────────

def route_after_grading(state: RAGState) -> str:
    """文档评分后路由：相关 → 生成；不相关 → 重写查询（最多重试 2 次）"""
    if state["doc_relevance"] == "relevant":
        return "generator"
    if state.get("retry_count", 0) >= 2:
        console.print("  [red]达到最大重试次数，使用当前文档生成[/red]")
        return "generator"
    return "rewriter"


def route_after_hallucination_check(state: RAGState) -> str:
    """幻觉检查后路由：有依据 → 结束；有幻觉 → 重新生成"""
    if state["hallucination_check"] == "grounded":
        return END
    return "generator"


# ── 6. 组装图 ───────────────────────────────────────────────────
# Agentic RAG 的图结构（对比 Phase 2 的线性管道）：
#
# Phase 2:  query → retrieve → generate → return
#
# Phase 3:  query_analyzer → retriever → doc_grader
#                                           ↓
#                              relevant → generator → hallucination_check
#                              irrelevant → rewriter → retriever (循环)
#                                                        ↓
#                                              grounded → END
#                                              hallucinated → generator (重试)

graph = StateGraph(RAGState)

graph.add_node("analyzer", query_analyzer)
graph.add_node("retriever", retriever)
graph.add_node("grader", doc_grader)
graph.add_node("generator", generator)
graph.add_node("checker", hallucination_checker)
graph.add_node("rewriter", query_rewriter)

graph.add_edge(START, "analyzer")
graph.add_edge("analyzer", "retriever")
graph.add_edge("retriever", "grader")
graph.add_conditional_edges("grader", route_after_grading, {
    "generator": "generator",
    "rewriter": "rewriter",
})
graph.add_edge("rewriter", "retriever")
graph.add_edge("generator", "checker")
graph.add_conditional_edges("checker", route_after_hallucination_check, {
    END: END,
    "generator": "generator",
})

app = graph.compile()


# ── 7. 运行演示 ─────────────────────────────────────────────────

def run_demo():
    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[red]请设置 DEEPSEEK_API_KEY 环境变量[/red]")
        console.print("[dim]cp .env.example .env 并填入你的 API Key[/dim]")
        return
    console.print(Panel(
        "[bold]LangGraph Agentic RAG[/bold]\n"
        "从 Phase 2 的线性 RAG 管道到自适应检索-生成图\n"
        "查询改写 → 检索 → 评分 → 生成 → 幻觉检查",
        title="04 Agentic RAG",
        border_style="blue",
    ))

    questions = [
        "LangGraph 和 CrewAI 有什么区别？",
        "如何防御 Agent 的 Prompt 注入攻击？",
        "RAG 系统的评估指标有哪些？",
    ]

    for q in questions:
        console.print(f"\n[bold yellow]问题: {q}[/bold yellow]")
        result = app.invoke({
            "question": q,
            "rewritten_query": "",
            "documents": [],
            "doc_relevance": "",
            "answer": "",
            "hallucination_check": "",
            "retry_count": 0,
        })
        console.print(Panel(result["answer"][:300], title="回答", border_style="green"))

    # 设计对比
    console.print(Panel(
        "[bold]线性 RAG vs Agentic RAG[/bold]\n\n"
        "Phase 2 线性管道:\n"
        "  query → retrieve → generate → return\n"
        "  问题：检索差就生成差，没有自我修正能力\n\n"
        "Phase 3 自适应图:\n"
        "  query → analyze → retrieve → grade → generate → check\n"
        "                       ↑                    ↓\n"
        "                    rewrite ← (不相关)   (有幻觉) → regenerate\n\n"
        "关键改进：\n"
        "  1. 查询改写 — 提高检索质量\n"
        "  2. 文档评分 — 不合格就重试\n"
        "  3. 幻觉检查 — 确保回答有依据\n"
        "  4. 自适应循环 — 失败不终止，而是换路径重试",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
