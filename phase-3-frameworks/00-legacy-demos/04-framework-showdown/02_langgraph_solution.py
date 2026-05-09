"""
02_langgraph_solution.py — 基准任务 LangGraph 实现：图编排范式

设计思考：
LangGraph 处理这个"研究→分析→报告"任务的思路是：
把工作流建模为有向图，每个阶段是一个节点，阶段间的过渡是边。

图结构：
    START → planner → [for each sub_topic] → researcher → analyzer → END
                                    ↑                         ↓
                                    └── synthesizer ←─────────┘

核心设计决策：
1. 状态设计：TypedDict State 作为节点间的数据契约
2. 控制流：条件边决定是否继续处理下一个子主题
3. 子图：每个子主题的研究可以是一个子图

与 CrewAI 方案的核心差异：
- LangGraph: 你定义图结构，框架执行图
- CrewAI: 你定义角色，框架协调

这导致代码结构完全不同：
- LangGraph: 节点函数 + 路由函数 + 图组装（工程师视角）
- CrewAI: Agent 角色 + Task 描述 + Crew 组装（管理者视角）

运行方式：
    cp ../01-langgraph-deep-dive/.env.example .env  # 填入 DEEPSEEK_API_KEY
    pip install langgraph langchain langchain-openai python-dotenv rich
    python 02_langgraph_solution.py
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from typing import Annotated, TypedDict

# 添加父目录到 path，以便引用共享定义
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_task_def = importlib.import_module("01_task_definition")
BenchmarkTask = _task_def.BenchmarkTask
EvaluationMetrics = _task_def.EvaluationMetrics
search_knowledge = _task_def.search_knowledge
KNOWLEDGE_BASE = _task_def.KNOWLEDGE_BASE

load_dotenv()
console = Console()


# ── 1. 状态定义 ─────────────────────────────────────────────────
# LangGraph 的核心：显式定义状态 schema
# 这就是"状态优先"设计——所有节点读写的契约在一处定义

class ResearchState(TypedDict):
    task: BenchmarkTask
    sub_topics: list[str]     # 待研究的子主题
    current_index: int        # 当前处理的子主题索引
    research_results: list[str]  # 各子主题的研究结果
    analysis: str             # 综合分析
    report: str               # 最终报告
    token_usage: list[dict]   # 每次 LLM 调用的 token 统计
    search_count: int         # 搜索调用计数
    llm_call_count: int       # LLM 调用计数


# ── 2. LLM 初始化 ──────────────────────────────────────────────

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3,
)


# ── 3. 节点函数 ─────────────────────────────────────────────────
# 每个节点对应研究过程的一个阶段
# 对比 CrewAI：这些不是"角色"，而是"处理步骤"

def planner_node(state: ResearchState) -> dict:
    """规划节点：拆解主题为子问题"""
    task = state["task"]
    console.print(f"\n  [cyan][Planner] 拆解研究主题...[/cyan]")

    response = llm.invoke([
        SystemMessage(content="你是研究规划专家。将研究主题拆解为 4-6 个具体子问题。只输出问题列表，每行一个。"),
        HumanMessage(content=f"研究主题: {task.title}\n\n任务: {task.topic}"),
    ])

    sub_topics = [s.strip() for s in response.content.strip().split("\n") if s.strip()][:6]

    console.print(Panel(
        "\n".join(f"  {i+1}. {s}" for i, s in enumerate(sub_topics)),
        title="研究计划",
        border_style="cyan",
    ))

    return {
        "sub_topics": sub_topics,
        "current_index": 0,
        "research_results": [],
        "token_usage": [{"node": "planner", "tokens": response.usage_metadata.get("total_tokens", 0)}],
        "llm_call_count": 1,
    }


def researcher_node(state: ResearchState) -> dict:
    """研究节点：搜索 + 分析一个子主题"""
    idx = state["current_index"]
    sub_topics = state["sub_topics"]

    if idx >= len(sub_topics):
        return {}

    topic = sub_topics[idx]
    console.print(f"\n  [green][Researcher] 研究子主题 {idx+1}/{len(sub_topics)}: {topic[:60]}...[/green]")

    # 搜索知识库
    search_results = search_knowledge(topic, top_k=3)
    context = "\n\n".join(
        f"[来源: {r['topic']}]\n{r['content']}" for r in search_results
    )

    # LLM 分析
    response = llm.invoke([
        SystemMessage(content=(
            "你是技术研究员。基于提供的参考资料，分析子问题。"
            "输出 150 字以内的分析摘要。如果有多个观点，列出关键差异。"
        )),
        HumanMessage(content=f"子问题: {topic}\n\n参考资料:\n{context}"),
    ])

    results = list(state.get("research_results", []))
    results.append(response.content.strip())

    token_usage = list(state.get("token_usage", []))
    token_usage.append({
        "node": f"researcher_{idx+1}",
        "tokens": response.usage_metadata.get("total_tokens", 0),
    })

    console.print(f"  [dim]研究结果: {response.content[:120]}...[/dim]")

    return {
        "research_results": results,
        "current_index": idx + 1,
        "token_usage": token_usage,
        "search_count": state.get("search_count", 0) + len(search_results),
        "llm_call_count": state.get("llm_call_count", 0) + 1,
    }


def analyzer_node(state: ResearchState) -> dict:
    """分析节点：综合所有研究结果，提炼关键洞察"""
    console.print(f"\n  [yellow][Analyzer] 综合分析...[/yellow]")

    results = state.get("research_results", [])
    combined = "\n\n---\n\n".join(
        f"子主题 {i+1}:\n{r}" for i, r in enumerate(results)
    )

    response = llm.invoke([
        SystemMessage(content=(
            "你是资深技术分析师。基于所有子主题的研究结果，提炼 3-5 个关键洞察。"
            "每个洞察用一句话概括，然后 50 字解释。输出格式：\n"
            "- 洞察 1: ...\n- 洞察 2: ..."
        )),
        HumanMessage(content=f"研究摘要:\n{combined}"),
    ])

    token_usage = list(state.get("token_usage", []))
    token_usage.append({"node": "analyzer", "tokens": response.usage_metadata.get("total_tokens", 0)})

    return {
        "analysis": response.content.strip(),
        "token_usage": token_usage,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
    }


def synthesizer_node(state: ResearchState) -> dict:
    """综合节点：生成最终报告"""
    console.print(f"\n  [magenta][Synthesizer] 生成最终报告...[/magenta]")

    task = state["task"]
    results = state.get("research_results", [])
    analysis = state.get("analysis", "")

    research_combined = "\n\n".join(
        f"### 子主题 {i+1}\n{r}" for i, r in enumerate(results)
    )

    response = llm.invoke([
        SystemMessage(content=(
            "你是技术报告撰写专家。基于研究结果和分析，生成一份结构化的研究报告。"
            "使用 Markdown 格式。包含：摘要、框架分析、架构对比表、选型建议、结论。"
            "报告控制在 800 字以内。"
        )),
        HumanMessage(content=(
            f"报告主题: {task.title}\n\n"
            f"研究结果:\n{research_combined}\n\n"
            f"综合分析:\n{analysis}\n\n"
            f"输出格式:\n{task.output_format}"
        )),
    ])

    token_usage = list(state.get("token_usage", []))
    token_usage.append({"node": "synthesizer", "tokens": response.usage_metadata.get("total_tokens", 0)})

    return {
        "report": response.content.strip(),
        "token_usage": token_usage,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
    }


# ── 4. 路由函数 ─────────────────────────────────────────────────
# LangGraph 的精髓：显式的路由逻辑
# 对比 CrewAI：CrewAI 不需要路由——框架自动按顺序推进

def after_researcher(state: ResearchState) -> str:
    """研究完一个子主题后：还有下一个 → 继续；全部完成 → 分析"""
    if state["current_index"] < len(state["sub_topics"]):
        return "researcher"
    return "analyzer"


# ── 5. 组装图 ───────────────────────────────────────────────────
# 这就是 LangGraph 的"声明式编排"：
#   节点 = 处理步骤
#   边 = 步骤间的数据流
#   条件边 = 分支逻辑

graph = StateGraph(ResearchState)

graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("analyzer", analyzer_node)
graph.add_node("synthesizer", synthesizer_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "researcher")
graph.add_conditional_edges("researcher", after_researcher, {
    "researcher": "researcher",
    "analyzer": "analyzer",
})
graph.add_edge("analyzer", "synthesizer")
graph.add_edge("synthesizer", END)

app = graph.compile()


# ── 6. 运行 ─────────────────────────────────────────────────────

def run_benchmark() -> tuple[str, EvaluationMetrics]:
    task = BenchmarkTask()
    metrics = EvaluationMetrics()

    t0 = time.perf_counter()

    result = app.invoke({
        "task": task,
        "sub_topics": [],
        "current_index": 0,
        "research_results": [],
        "analysis": "",
        "report": "",
        "token_usage": [],
        "search_count": 0,
        "llm_call_count": 0,
    })

    elapsed = time.perf_counter() - t0
    total_tokens = sum(t.get("tokens", 0) for t in result.get("token_usage", []))

    metrics = EvaluationMetrics(
        total_tokens=total_tokens,
        total_latency_seconds=elapsed,
        num_llm_calls=result.get("llm_call_count", 0),
        num_tool_calls=result.get("search_count", 0),
        output_length=len(result.get("report", "")),
        num_sections=result.get("report", "").count("## "),
        source_citations=sum(1 for r in result.get("research_results", []) if "[来源:" in r),
    )

    return result.get("report", ""), metrics


def run_demo():
    console.print(Panel(
        "[bold]LangGraph 方案：图编排范式[/bold]\n"
        "planner → researcher (循环) → analyzer → synthesizer\n"
        "显式状态 + 声明式路由 + 条件边循环",
        title="02 LangGraph Solution",
        border_style="blue",
    ))

    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[red]需要设置 DEEPSEEK_API_KEY 环境变量[/red]")
        return

    # 展示图结构
    console.print(Panel(
        "START → planner → researcher ←─┐\n"
        "                      ↓          │ (仍有子主题)\n"
        "                    analyzer ────┘\n"
        "                      ↓\n"
        "                  synthesizer\n"
        "                      ↓\n"
        "                     END",
        title="图结构（LangGraph 的声明式控制流）",
        border_style="cyan",
    ))

    console.print("\n[bold yellow]开始执行研究任务...[/bold yellow]")
    report, metrics = run_benchmark()

    console.print(Panel(report[:800], title="研究报告", border_style="green"))
    console.print(metrics.to_table())

    console.print(Panel(
        "[bold]LangGraph 方案的架构特点[/bold]\n\n"
        "代码组织:\n"
        "  - State schema: 定义所有节点共享的数据契约\n"
        "  - 4 个节点函数: 每个是纯函数（State → partial State）\n"
        "  - 1 个路由函数: 显式的条件逻辑\n"
        "  - 图组装: 声明式地描述整个工作流\n\n"
        "设计优势:\n"
        "  1. 图结构可视化（上面的 ASCII 图就是图本身）\n"
        "  2. 每个节点独立可测试\n"
        "  3. 可以轻松加条件分支（如检索质量不合格→重试）\n"
        "  4. 可以用 checkpointer 持久化中间状态\n\n"
        "设计代价:\n"
        "  1. 需要预先设计 State schema\n"
        "  2. 需要理解图执行模型（节点、边、条件边）\n"
        "  3. 对简单流程来说过度设计",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
