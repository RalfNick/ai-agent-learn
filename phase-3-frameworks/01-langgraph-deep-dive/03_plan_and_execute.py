"""
03_plan_and_execute.py — LangGraph Plan-and-Execute：任务拆解与自适应执行

设计思考：
Phase 1 的 ExecutionEngine 有一个 planning_interval 参数：
每隔 N 步暂停，让 LLM 反思并调整计划。这是"规划"的雏形。

LangGraph 的 Plan-and-Execute 把这个思路推到极致：
- Planner 节点：拆解复杂任务为子步骤
- Executor 节点：逐步执行
- Reflector 节点：评估执行结果，决定是否需要重新规划

这是处理复杂任务的核心模式：
  简单任务 → 单次 ReAct 循环就够（01 的模式）
  复杂任务 → 先规划再执行，执行中可以修正计划

关键设计选择：
- 规划和执行分离 → 可以用不同的模型（规划用强模型，执行用快模型）
- 反思节点 → 失败不是终止，而是触发重新规划
- 步骤状态追踪 → 知道哪些做了、哪些没做

运行方式：
    python 03_plan_and_execute.py
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()


# ── 1. 状态定义 ─────────────────────────────────────────────────
# Plan-and-Execute 的状态比简单 ReAct 复杂：
# 需要追踪计划、当前步骤、执行结果、是否需要重规划

class PlanExecuteState(TypedDict):
    task: str                    # 原始任务
    plan: list[str]              # 当前计划（步骤列表）
    current_step: int            # 当前执行到第几步
    step_results: list[str]      # 每步的执行结果
    reflection: str              # 反思内容
    needs_replan: bool           # 是否需要重新规划
    final_answer: str            # 最终答案


# ── 2. LLM 初始化 ──────────────────────────────────────────────

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3,
)


# ── 3. 节点函数 ─────────────────────────────────────────────────

def planner(state: PlanExecuteState) -> dict:
    """规划节点：将复杂任务拆解为可执行的步骤列表"""
    task = state["task"]
    existing_results = state.get("step_results", [])

    if existing_results:
        context = f"已完成的步骤结果:\n" + "\n".join(
            f"  {i+1}. {r}" for i, r in enumerate(existing_results)
        )
        prompt = (
            f"任务: {task}\n\n{context}\n\n"
            f"反思: {state.get('reflection', '')}\n\n"
            "基于已有结果和反思，重新规划剩余步骤。"
            "只输出步骤列表，每行一个步骤，不要编号。"
        )
    else:
        prompt = (
            f"任务: {task}\n\n"
            "将这个任务拆解为 3-5 个具体的执行步骤。"
            "每个步骤应该是独立可执行的。"
            "只输出步骤列表，每行一个步骤，不要编号。"
        )

    response = llm.invoke([
        SystemMessage(content="你是一个任务规划专家。将复杂任务拆解为清晰、可执行的步骤。"),
        HumanMessage(content=prompt),
    ])

    steps = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
    console.print(Panel(
        "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)),
        title="📋 规划结果",
        border_style="cyan",
    ))
    return {"plan": steps, "current_step": 0, "needs_replan": False}


def executor(state: PlanExecuteState) -> dict:
    """执行节点：执行当前步骤"""
    step_idx = state["current_step"]
    plan = state["plan"]

    if step_idx >= len(plan):
        return {"step_results": state.get("step_results", [])}

    current_task = plan[step_idx]
    context = ""
    if state.get("step_results"):
        context = "前序步骤结果:\n" + "\n".join(
            f"  - {r}" for r in state["step_results"]
        )

    response = llm.invoke([
        SystemMessage(content="你是一个任务执行者。简洁地完成给定的步骤，输出执行结果。"),
        HumanMessage(content=f"执行步骤: {current_task}\n\n{context}"),
    ])

    result = response.content.strip()
    results = list(state.get("step_results", []))
    results.append(result)

    console.print(f"  [green]✓ 步骤 {step_idx + 1}:[/green] {current_task[:50]}...")
    return {"step_results": results, "current_step": step_idx + 1}


def reflector(state: PlanExecuteState) -> dict:
    """反思节点：评估执行进度，决定是否需要重新规划"""
    task = state["task"]
    results = state.get("step_results", [])
    plan = state["plan"]

    response = llm.invoke([
        SystemMessage(content=(
            "你是一个质量评估专家。评估任务执行情况。\n"
            "回答格式：第一行写 COMPLETE 或 REPLAN\n"
            "第二行写简短理由。"
        )),
        HumanMessage(content=(
            f"原始任务: {task}\n\n"
            f"计划步骤: {len(plan)} 步\n"
            f"已执行: {len(results)} 步\n\n"
            f"执行结果摘要:\n" + "\n".join(f"  {i+1}. {r[:100]}" for i, r in enumerate(results))
        )),
    ])

    content = response.content.strip()
    needs_replan = "REPLAN" in content.split("\n")[0].upper()
    reflection = content.split("\n", 1)[-1].strip() if "\n" in content else content

    if needs_replan:
        console.print(f"  [yellow]↻ 需要重新规划:[/yellow] {reflection[:80]}")
    else:
        console.print(f"  [green]✓ 执行完成[/green]")

    return {"reflection": reflection, "needs_replan": needs_replan}


def synthesizer(state: PlanExecuteState) -> dict:
    """综合节点：汇总所有步骤结果，生成最终答案"""
    response = llm.invoke([
        SystemMessage(content="综合所有步骤的执行结果，生成一个完整、连贯的最终答案。"),
        HumanMessage(content=(
            f"任务: {state['task']}\n\n"
            f"各步骤结果:\n" + "\n".join(
                f"  {i+1}. {r}" for i, r in enumerate(state["step_results"])
            )
        )),
    ])
    return {"final_answer": response.content}


# ── 4. 路由函数 ─────────────────────────────────────────────────

def should_execute_or_reflect(state: PlanExecuteState) -> str:
    """执行完当前步骤后：还有步骤 → 继续执行；全部完成 → 反思"""
    if state["current_step"] < len(state["plan"]):
        return "executor"
    return "reflector"


def should_replan_or_finish(state: PlanExecuteState) -> str:
    """反思后：需要重规划 → 回到 planner；否则 → 综合输出"""
    if state.get("needs_replan"):
        return "planner"
    return "synthesizer"


# ── 5. 组装图 ───────────────────────────────────────────────────
# Plan-and-Execute 的图结构：
#   planner → executor (循环执行所有步骤) → reflector → (replan | synthesizer)
#                                                          ↑         ↓
#                                                       planner     END

graph = StateGraph(PlanExecuteState)

graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("reflector", reflector)
graph.add_node("synthesizer", synthesizer)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_execute_or_reflect, {
    "executor": "executor",
    "reflector": "reflector",
})
graph.add_conditional_edges("reflector", should_replan_or_finish, {
    "planner": "planner",
    "synthesizer": "synthesizer",
})
graph.add_edge("synthesizer", END)

app = graph.compile()


# ── 6. 运行演示 ─────────────────────────────────────────────────

def run_demo():
    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[red]请设置 DEEPSEEK_API_KEY 环境变量[/red]")
        console.print("[dim]cp .env.example .env 并填入你的 API Key[/dim]")
        return
    console.print(Panel(
        "[bold]LangGraph Plan-and-Execute[/bold]\n"
        "复杂任务处理的核心模式：规划 → 执行 → 反思 → 重规划\n"
        "对比 Phase 1 的 planning_interval：从定时反思到自适应反思",
        title="03 Plan-and-Execute",
        border_style="blue",
    ))

    task = "分析 Python 和 Rust 在 AI 开发领域的优劣势，给出技术选型建议"

    console.print(f"\n[bold]任务:[/bold] {task}\n")

    result = app.invoke({
        "task": task,
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "reflection": "",
        "needs_replan": False,
        "final_answer": "",
    })

    console.print(Panel(result["final_answer"], title="最终答案", border_style="green"))

    # 设计对比
    console.print(Panel(
        "[bold]Plan-and-Execute 的设计哲学[/bold]\n\n"
        "Phase 1 的 planning_interval:\n"
        "  每 N 步强制反思一次（固定间隔，不管需不需要）\n\n"
        "LangGraph 的 Plan-and-Execute:\n"
        "  规划 → 全部执行完 → 反思 → 按需重规划（自适应）\n\n"
        "核心优势：\n"
        "  1. 分离关注点 — 规划和执行可以用不同模型\n"
        "  2. 容错 — 某步失败不终止，反思后重新规划\n"
        "  3. 可观测 — 每步结果独立记录，便于调试\n"
        "  4. 可扩展 — 加入人类审核只需在 reflector 后加 interrupt\n\n"
        "适用场景：\n"
        "  - 多步骤研究任务\n"
        "  - 代码生成 + 测试 + 修复循环\n"
        "  - 数据处理管道",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
