"""
01_crew_basics.py — CrewAI 基础：角色、任务与团队

设计思考：
LangGraph 用"图"来建模 Agent 工作流 —— 节点是函数，边是控制流。
CrewAI 用"组织"来建模 —— Agent 是有角色的人，Task 是工作任务，Crew 是团队。

这是两种完全不同的抽象隐喻：
- LangGraph: Agent = Node + State（工程师视角：数据流和状态机）
- CrewAI:   Agent = Role + Goal + Backstory（管理者视角：角色和职责）

CrewAI 的核心洞察：
  定义好"谁做什么"比定义"数据怎么流"更直觉。
  对于多角色协作场景，组织隐喻比图隐喻更自然。

但这也带来 tradeoff：
  + 快速原型：定义角色就能跑，不需要画图
  + 可读性：非技术人员也能理解"市场分析师 → 报告撰写者"
  - 控制力：你不能精确控制执行路径（框架决定）
  - 可调试性：出问题时不如 LangGraph 的图结构清晰

运行方式：
    cp .env.example .env  # 填入 API Key
    pip install -r requirements.txt
    python 01_crew_basics.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()


# ── 1. 定义 LLM ────────────────────────────────────────────────
# CrewAI 通过 LLM 类适配不同模型提供商

llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# ── 2. 定义 Agent（角色）────────────────────────────────────────
# CrewAI 的 Agent 三要素：Role + Goal + Backstory
# 这不是随意的文本 —— 它们直接影响 LLM 的行为：
#   role → 决定 Agent 的专业领域
#   goal → 决定 Agent 的优化方向
#   backstory → 提供上下文，影响回答风格和深度

researcher = Agent(
    role="技术研究员",
    goal="深入研究技术主题，提供准确、全面的信息",
    backstory=(
        "你是一位资深技术研究员，擅长快速理解新技术并提炼核心要点。"
        "你的研究以准确性和深度著称，总能找到别人忽略的关键细节。"
    ),
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="技术写作者",
    goal="将复杂技术内容转化为清晰、有洞察力的文章",
    backstory=(
        "你是一位技术博客作者，擅长用简洁的语言解释复杂概念。"
        "你的文章既有技术深度，又能让非专家读者理解核心思想。"
    ),
    llm=llm,
    verbose=True,
)


# ── 3. 定义 Task（任务）─────────────────────────────────────────
# Task 的关键字段：
#   description → 具体要做什么（越具体越好）
#   expected_output → 期望的输出格式（约束 LLM 输出）
#   agent → 分配给哪个 Agent

research_task = Task(
    description=(
        "研究 LangGraph 框架的核心设计理念和技术架构。"
        "重点关注：1) 状态图模型 2) 与传统 Agent 框架的区别 3) 适用场景"
    ),
    expected_output="一份 300 字以内的技术研究摘要，包含核心要点和关键数据",
    agent=researcher,
)

writing_task = Task(
    description=(
        "基于研究摘要，撰写一段面向开发者的技术介绍。"
        "要求：通俗易懂，有具体例子，突出实用价值。"
    ),
    expected_output="一段 200 字以内的技术介绍文案，适合发布在技术社区",
    agent=writer,
)


# ── 4. 组装 Crew（团队）─────────────────────────────────────────
# Crew 把 Agent 和 Task 组合在一起
# Process.sequential = 按顺序执行（Task 1 完成后 Task 2 才开始）
# 前一个 Task 的输出自动成为后一个 Task 的上下文

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # 顺序执行：研究 → 写作
    verbose=True,
)


# ── 5. 运行演示 ─────────────────────────────────────────────────

def run_demo():
    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[red]请设置 DEEPSEEK_API_KEY 环境变量[/red]")
        console.print("[dim]cp .env.example .env 并填入你的 API Key[/dim]")
        return
    console.print(Panel(
        "[bold]CrewAI 基础：角色、任务与团队[/bold]\n"
        "用「组织隐喻」建模多 Agent 协作\n"
        "研究员 → 写作者（Sequential 流程）",
        title="01 Crew Basics",
        border_style="blue",
    ))

    # 展示 Crew 结构
    table = Table(title="Crew 结构")
    table.add_column("Agent", style="cyan")
    table.add_column("Role", style="green")
    table.add_column("Task", style="yellow")
    table.add_row("researcher", "技术研究员", "研究 LangGraph 核心设计")
    table.add_row("writer", "技术写作者", "撰写技术介绍文案")
    console.print(table)

    console.print("\n[bold yellow]开始执行 Crew...[/bold yellow]\n")

    result = crew.kickoff()

    console.print(Panel(str(result), title="Crew 最终输出", border_style="green"))

    # 设计对比
    console.print(Panel(
        "[bold]CrewAI vs LangGraph 设计对比[/bold]\n\n"
        "LangGraph 定义工作流：\n"
        "  graph.add_node('research', research_fn)\n"
        "  graph.add_node('write', write_fn)\n"
        "  graph.add_edge('research', 'write')\n"
        "  → 你控制数据如何流动\n\n"
        "CrewAI 定义组织：\n"
        "  researcher = Agent(role='研究员', goal='...')\n"
        "  writer = Agent(role='写作者', goal='...')\n"
        "  crew = Crew(agents=[...], process=sequential)\n"
        "  → 你定义谁做什么，框架决定怎么协调\n\n"
        "Tradeoff:\n"
        "  CrewAI: 快速原型 ✓ | 精确控制 ✗\n"
        "  LangGraph: 精确控制 ✓ | 快速原型 ✗",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
