"""
02_hierarchical_delegation.py — CrewAI 层级委派：Manager Agent 自动调度

设计思考：
01 中的 Sequential 流程是"流水线"：A 做完给 B，B 做完给 C。
Hierarchical 流程是"经理制"：Manager Agent 看全局，动态分配任务。

这对应 Phase 1 中手写的 SupervisorOrchestrator：
- Phase 1: Supervisor 用 LLM 决定下一个执行的 Agent（显式路由）
- CrewAI: Manager 自动协调，开发者不需要写路由逻辑（隐式路由）
- LangGraph: 用条件边实现路由（显式但声明式）

三种方式的 tradeoff：
  Phase 1 手写: 完全控制，但代码量大
  LangGraph:   声明式控制，灵活但需要设计图
  CrewAI:      零控制代码，快但不透明

运行方式：
    python 02_hierarchical_delegation.py
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


# ── 1. 定义专家 Agent ──────────────────────────────────────────
# Hierarchical 模式下，Agent 不需要预先分配 Task
# Manager 会根据任务需求动态选择合适的 Agent

coder = Agent(
    role="Python 开发工程师",
    goal="编写高质量、可维护的 Python 代码",
    backstory=(
        "你是一位有 10 年经验的 Python 开发者，精通设计模式和最佳实践。"
        "你写的代码简洁、有类型注解、有适当的错误处理。"
    ),
    llm=llm,
    allow_delegation=False,  # 工人不委派，只执行
)

reviewer = Agent(
    role="代码审查专家",
    goal="发现代码中的问题并提出改进建议",
    backstory=(
        "你是一位严格的代码审查者，关注安全性、性能和可读性。"
        "你总能发现潜在的 bug 和设计缺陷。"
    ),
    llm=llm,
    allow_delegation=False,
)

architect = Agent(
    role="软件架构师",
    goal="设计清晰、可扩展的系统架构",
    backstory=(
        "你是一位资深架构师，擅长在简洁性和灵活性之间找到平衡。"
        "你的设计决策总是有充分的技术理由。"
    ),
    llm=llm,
    allow_delegation=True,  # 架构师可以委派子任务
)


# ── 2. 定义任务 ─────────────────────────────────────────────────
# Hierarchical 模式下，Task 可以不指定 agent
# Manager 会根据 Agent 的 role/goal 自动匹配

design_task = Task(
    description=(
        "设计一个简单的 URL 短链接服务的架构。"
        "包括：核心组件、数据模型、API 接口设计。"
        "要求简洁实用，适合小团队快速实现。"
    ),
    expected_output="一份简洁的架构设计文档，包含组件图、数据模型和 API 列表",
)

implement_task = Task(
    description=(
        "根据架构设计，编写 URL 短链接服务的核心代码。"
        "包括：短链生成算法、存储接口、API 路由。"
        "使用 Python，代码要有类型注解。"
    ),
    expected_output="可运行的 Python 代码，包含核心业务逻辑",
)

review_task = Task(
    description=(
        "审查架构设计和代码实现。"
        "检查：安全性、性能瓶颈、代码质量、潜在问题。"
        "给出具体的改进建议。"
    ),
    expected_output="代码审查报告，包含问题列表和改进建议",
)


# ── 3. 组装 Hierarchical Crew ──────────────────────────────────
# Process.hierarchical 会自动创建一个 Manager Agent
# Manager 负责：分析任务 → 选择 Agent → 分配 → 验证结果

crew = Crew(
    agents=[architect, coder, reviewer],
    tasks=[design_task, implement_task, review_task],
    process=Process.hierarchical,
    manager_llm=llm,
    verbose=True,
)


# ── 4. 运行演示 ─────────────────────────────────────────────────

def run_demo():
    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[red]请设置 DEEPSEEK_API_KEY 环境变量[/red]")
        console.print("[dim]cp .env.example .env 并填入你的 API Key[/dim]")
        return
    console.print(Panel(
        "[bold]CrewAI 层级委派[/bold]\n"
        "Manager Agent 自动调度：架构师 + 开发者 + 审查者\n"
        "对比 Phase 1 的 SupervisorOrchestrator",
        title="02 Hierarchical Delegation",
        border_style="blue",
    ))

    table = Table(title="团队结构（Hierarchical）")
    table.add_column("角色", style="cyan")
    table.add_column("可委派", style="green")
    table.add_column("职责")
    table.add_row("Manager (自动)", "✓", "分析任务、选择 Agent、验证结果")
    table.add_row("架构师", "✓", "系统设计")
    table.add_row("开发者", "✗", "代码实现")
    table.add_row("审查者", "✗", "代码审查")
    console.print(table)

    console.print("\n[bold yellow]开始执行 Hierarchical Crew...[/bold yellow]\n")

    result = crew.kickoff()

    console.print(Panel(str(result)[:500], title="Crew 最终输出", border_style="green"))

    console.print(Panel(
        "[bold]三种 Supervisor 模式对比[/bold]\n\n"
        "Phase 1 手写 SupervisorOrchestrator:\n"
        "  supervisor.route(task) → 用 LLM 选择下一个 Agent\n"
        "  你写路由 prompt，你控制一切\n\n"
        "LangGraph 条件边:\n"
        "  graph.add_conditional_edges('supervisor', route_fn)\n"
        "  路由逻辑是显式函数，可测试、可调试\n\n"
        "CrewAI Hierarchical:\n"
        "  crew = Crew(process=Process.hierarchical)\n"
        "  Manager 自动生成，路由逻辑隐式\n\n"
        "选择依据:\n"
        "  需要精确控制 → LangGraph\n"
        "  快速验证想法 → CrewAI\n"
        "  学习原理 → Phase 1 手写",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
