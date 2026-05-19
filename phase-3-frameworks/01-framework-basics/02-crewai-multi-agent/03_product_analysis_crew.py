"""
03_product_analysis_crew.py — CrewAI 实战：4 角色产品分析团队

设计思考：
前两个文件展示了 CrewAI 的基础和层级模式。
这个文件展示 CrewAI 真正的甜蜜点：快速原型化多角色工作流。

场景：输入一个产品创意，4 个角色协作产出完整的分析报告。
- 市场研究员：分析市场规模和趋势
- 竞品分析师：分析竞争格局
- 技术评估师：评估技术可行性
- 报告撰写者：综合所有分析，产出最终报告

这个场景用 LangGraph 也能做，但需要：
1. 定义 State schema（4 个角色的输出字段）
2. 写 4 个节点函数
3. 设计边和路由
4. 处理上下文传递

用 CrewAI：定义 4 个 Agent + 4 个 Task + 1 个 Crew，完事。
这就是"组织隐喻"的威力 —— 对于角色明确的协作场景，CrewAI 的表达力更强。

运行方式：
    python 03_product_analysis_crew.py
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

llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# ── 1. 定义 4 个专家角色 ───────────────────────────────────────

market_researcher = Agent(
    role="市场研究员",
    goal="分析目标市场的规模、增长趋势和用户需求",
    backstory=(
        "你是一位资深市场研究员，擅长从公开数据中提炼市场洞察。"
        "你的分析总是基于数据，结论清晰有力。"
        "你特别关注市场规模（TAM/SAM/SOM）和增长驱动因素。"
    ),
    llm=llm,
    verbose=True,
)

competitor_analyst = Agent(
    role="竞品分析师",
    goal="全面分析竞争格局，找出差异化机会",
    backstory=(
        "你是一位竞品情报专家，擅长分析竞争对手的产品策略和市场定位。"
        "你总能找到市场空白和差异化切入点。"
        "你的分析框架包括：直接竞品、间接竞品、替代方案。"
    ),
    llm=llm,
    verbose=True,
)

tech_evaluator = Agent(
    role="技术评估师",
    goal="评估产品的技术可行性和实现路径",
    backstory=(
        "你是一位全栈技术专家，擅长评估技术方案的可行性和风险。"
        "你关注：技术栈选择、开发周期、技术壁垒、可扩展性。"
        "你的建议总是务实的，考虑团队规模和资源约束。"
    ),
    llm=llm,
    verbose=True,
)

report_writer = Agent(
    role="分析报告撰写者",
    goal="综合所有分析，产出结构清晰、有洞察力的产品分析报告",
    backstory=(
        "你是一位商业分析报告专家，擅长将多维度分析综合为可执行的建议。"
        "你的报告结构清晰，结论明确，总是以「行动建议」结尾。"
    ),
    llm=llm,
    verbose=True,
)


# ── 2. 定义任务链 ──────────────────────────────────────────────
# Sequential 流程中，前一个 Task 的输出自动传递给后一个 Task
# 这就是 CrewAI 的"上下文传递"机制 —— 不需要手动管理状态

market_task = Task(
    description=(
        "分析「{product_idea}」的目标市场。"
        "包括：1) 市场规模估算 2) 目标用户画像 3) 增长趋势 4) 关键需求痛点"
    ),
    expected_output="市场分析摘要（200字以内），包含关键数据点",
    agent=market_researcher,
)

competitor_task = Task(
    description=(
        "基于市场分析，分析「{product_idea}」的竞争格局。"
        "包括：1) 主要竞品列表 2) 各竞品优劣势 3) 市场空白 4) 差异化机会"
    ),
    expected_output="竞品分析摘要（200字以内），包含差异化建议",
    agent=competitor_analyst,
)

tech_task = Task(
    description=(
        "评估「{product_idea}」的技术可行性。"
        "包括：1) 推荐技术栈 2) 核心技术挑战 3) MVP 开发周期估算 4) 技术风险"
    ),
    expected_output="技术评估摘要（200字以内），包含技术栈建议和时间估算",
    agent=tech_evaluator,
)

report_task = Task(
    description=(
        "综合市场分析、竞品分析和技术评估，撰写「{product_idea}」的产品分析报告。"
        "报告结构：1) 执行摘要 2) 市场机会 3) 竞争优势 4) 技术路径 5) 行动建议"
    ),
    expected_output="完整的产品分析报告（500字以内），以明确的行动建议结尾",
    agent=report_writer,
)


# ── 3. 组装 Crew ───────────────────────────────────────────────

product_crew = Crew(
    agents=[market_researcher, competitor_analyst, tech_evaluator, report_writer],
    tasks=[market_task, competitor_task, tech_task, report_task],
    process=Process.sequential,
    verbose=True,
)


# ── 4. 运行演示 ─────────────────────────────────────────────────

def run_demo():
    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[red]请设置 DEEPSEEK_API_KEY 环境变量[/red]")
        console.print("[dim]cp .env.example .env 并填入你的 API Key[/dim]")
        return
    console.print(Panel(
        "[bold]CrewAI 实战：产品分析团队[/bold]\n"
        "4 角色协作：市场研究 → 竞品分析 → 技术评估 → 综合报告\n"
        "展示 CrewAI 在多角色协作场景的表达力",
        title="03 Product Analysis Crew",
        border_style="blue",
    ))

    table = Table(title="产品分析团队")
    table.add_column("角色", style="cyan")
    table.add_column("职责", style="green")
    table.add_column("输出")
    table.add_row("市场研究员", "分析市场规模和趋势", "市场分析摘要")
    table.add_row("竞品分析师", "分析竞争格局", "竞品分析 + 差异化建议")
    table.add_row("技术评估师", "评估技术可行性", "技术栈 + 时间估算")
    table.add_row("报告撰写者", "综合产出报告", "完整产品分析报告")
    console.print(table)

    product_idea = "一个面向独立开发者的 AI Agent 开发平台，提供可视化工作流编排和一键部署"

    console.print(f"\n[bold]产品创意:[/bold] {product_idea}\n")
    console.print("[bold yellow]开始多角色协作分析...[/bold yellow]\n")

    result = product_crew.kickoff(inputs={"product_idea": product_idea})

    console.print(Panel(str(result), title="产品分析报告", border_style="green"))

    console.print(Panel(
        "[bold]CrewAI 的甜蜜点[/bold]\n\n"
        "这个 4 角色协作场景，用 CrewAI 只需要：\n"
        "  - 4 个 Agent 定义（角色 + 目标 + 背景）\n"
        "  - 4 个 Task 定义（描述 + 期望输出）\n"
        "  - 1 个 Crew 组装\n"
        "  总计 ~50 行核心代码\n\n"
        "同样的场景用 LangGraph 需要：\n"
        "  - State schema（定义所有中间数据）\n"
        "  - 4 个节点函数（手动管理上下文传递）\n"
        "  - 边和路由逻辑\n"
        "  总计 ~120 行核心代码\n\n"
        "结论：\n"
        "  角色明确 + 流程固定 → CrewAI 更高效\n"
        "  需要条件分支 + 循环重试 → LangGraph 更合适\n"
        "  两者不是替代关系，而是互补",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
