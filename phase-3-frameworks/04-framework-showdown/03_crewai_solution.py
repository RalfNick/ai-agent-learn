"""
03_crewai_solution.py — 基准任务 CrewAI 实现：角色协作范式

设计思考：
CrewAI 处理这个"研究→分析→报告"任务的思路是：
组建一个研究团队，每个 Agent 有明确的角色和职责，按流程协作。

团队结构：
    Research Planner → Topic Researcher → Insight Analyzer → Report Writer
    (规划者)           (研究者)           (分析者)          (撰写者)

这就是 CrewAI 的"组织隐喻"：
- 不需要定义 State schema（框架自动管理上下文传递）
- 不需要写路由函数（Sequential 流程自动按序执行）
- 不需要设计图结构（用角色和任务的语义来描述流程）

与 LangGraph 方案的核心差异：
同一个任务、同一个 LLM、同一个知识库——
LangGraph 用了 4 个节点函数 + 路由 + 图组装，
CrewAI 用了 4 个 Agent 角色 + 4 个 Task 定义 + Crew 组装。

代码量和代码结构的差异反映了两种哲学的根本分歧：
- LangGraph: 工程师视角——数据流和控制流是一等公民
- CrewAI: 管理者视角——角色和职责是一等公民

运行方式：
    cp ../02-crewai-multi-agent/.env.example .env  # 填入 DEEPSEEK_API_KEY
    pip install crewai crewai-tools python-dotenv rich
    python 03_crewai_solution.py
"""

from __future__ import annotations

import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_task_def = importlib.import_module("01_task_definition")
BenchmarkTask = _task_def.BenchmarkTask
EvaluationMetrics = _task_def.EvaluationMetrics
KNOWLEDGE_BASE = _task_def.KNOWLEDGE_BASE

load_dotenv()
console = Console()


# ── 1. 定义 LLM ─────────────────────────────────────────────────

llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# ── 2. 定义研究团队（4 个角色）─────────────────────────────────
# CrewAI 的 Agent 定义三要素：Role + Goal + Backstory
# 这不是装饰性文本——它们直接影响 LLM 的行为模式
# Backstory 提供上下文，影响回答风格和深度
# Goal 决定 Agent 优化的方向（LLM 会朝目标方向调整输出）

planner = Agent(
    role="研究规划师",
    goal="将研究主题拆解为具体的子问题，制定研究路线图",
    backstory=(
        "你是一位资深研究规划师，擅长将复杂主题拆解为可操作的子问题。"
        "你的规划总是有逻辑层次，从大到小、从抽象到具体。"
        f"你掌握以下知识领域: {', '.join(KNOWLEDGE_BASE.keys())}。"
    ),
    llm=llm,
    verbose=False,
)

researcher = Agent(
    role="技术研究员",
    goal="深入调研每个子问题，获取准确、全面的信息，提炼核心观点",
    backstory=(
        "你是一位严谨的技术研究员，擅长从参考资料中提取关键信息。"
        "你注重事实准确性，每个观点都有依据。"
        "你善于比较不同框架的设计哲学，找出根本差异而非表面特征。"
        f"你可查阅的知识库涵盖: {', '.join(KNOWLEDGE_BASE.keys())}。"
    ),
    llm=llm,
    verbose=False,
)

analyst = Agent(
    role="技术分析师",
    goal="综合研究成果，提炼关键洞察和对比分析",
    backstory=(
        "你是一位资深技术分析师，擅长从零散的研究结果中提炼模式。"
        "你能看到不同框架设计选择背后的哲学差异，而不是仅仅比较功能列表。"
        "你的分析总是有深度，能揭示表面之下的根本差异。"
    ),
    llm=llm,
    verbose=False,
)

writer = Agent(
    role="技术报告撰写者",
    goal="将分析结果转化为结构清晰、有说服力的研究报告",
    backstory=(
        "你是一位技术报告专家，擅长用清晰的结构和准确的语言传达复杂的技术分析。"
        "你的报告总是有明确的受众（本文面向企业开发者），"
        "有可执行建议（而非模糊的废话），"
        "有数据支撑（而非个人观点）。"
    ),
    llm=llm,
    verbose=False,
)


# ── 3. 定义任务链 ──────────────────────────────────────────────
# CrewAI 的 Task 定义：描述 + 期望输出 + 分配的 Agent
# 前一个 Task 的输出自动成为后一个 Task 的上下文输入
# 这就是 CrewAI 的"隐式状态传递"——不需要定义 State schema

plan_task = Task(
    description=(
        "研究主题: {topic}\n\n"
        "将该主题拆解为 4-6 个具体的子问题。每个子问题应该是独立可研究的。"
        "只输出子问题列表，每行一个，不要编号。"
    ),
    expected_output="4-6 个子问题列表，每行一个",
    agent=planner,
)

research_task = Task(
    description=(
        "基于研究规划，逐个分析每个子问题。\n\n"
        "可用参考资料:\n{knowledge_base}\n\n"
        "对每个子问题，给出 100-150 字的分析。\n"
        "重点关注各框架的设计哲学差异，而非表面功能对比。"
    ),
    expected_output="每个子问题的分析摘要，标注来源",
    agent=researcher,
)

analyze_task = Task(
    description=(
        "基于研究结果，进行综合对比分析。\n"
        "找出三个框架在最核心维度上的差异：控制流管理、多 Agent 协作、安全设计、学习曲线。\n"
        "提炼 3-5 个关键洞察，每个用一句话概括 + 简短解释。"
    ),
    expected_output="对比分析要点，包含关键洞察列表",
    agent=analyst,
)

write_task = Task(
    description=(
        "基于所有研究和分析结果，撰写一份完整的 {title}。\n\n"
        "报告结构:\n"
        "## 摘要\n"
        "- 核心结论一句话\n"
        "- 3-5 个关键发现\n\n"
        "## 各框架分析\n"
        "- 分别分析三个框架的核心设计和适用场景\n\n"
        "## 架构对比\n"
        "- 用表格比较关键维度\n\n"
        "## 选型建议\n"
        "- 按团队规模、任务复杂度、安全需求给出具体建议\n\n"
        "## 结论\n\n"
        "报告目标读者: 企业开发者\n"
        "报告长度: 600-800 字\n"
        "风格: 技术性强、有具体例子、结论可执行"
    ),
    expected_output="完整的 Markdown 格式研究报告",
    agent=writer,
)


# ── 4. 组装 Crew ──────────────────────────────────────────────
# Sequential 流程：按顺序执行，前一个 Task 的输出自动成为后一个的输入
# 对比 LangGraph: 不需要定义 State schema 和路由函数
# 对比 Claude SDK: 不需要手写循环和状态管理

research_crew = Crew(
    agents=[planner, researcher, analyst, writer],
    tasks=[plan_task, research_task, analyze_task, write_task],
    process=Process.sequential,
    verbose=False,
)


# ── 5. 运行 ─────────────────────────────────────────────────────

def run_benchmark() -> tuple[str, EvaluationMetrics]:
    task = BenchmarkTask()

    # 构建知识库字符串（CrewAI Task 的 description 支持 {variable} 模板）
    kb_str = "\n\n".join(
        f"### {topic}\n{content}" for topic, content in KNOWLEDGE_BASE.items()
    )

    t0 = time.perf_counter()

    result = research_crew.kickoff(inputs={
        "topic": task.topic,
        "title": task.title,
        "knowledge_base": kb_str,
    })

    elapsed = time.perf_counter() - t0
    output = str(result)

    metrics = EvaluationMetrics(
        total_tokens=0,  # CrewAI 不暴露 token 统计
        total_latency_seconds=elapsed,
        num_llm_calls=0,  # CrewAI 不暴露调用次数
        num_tool_calls=0,
        output_length=len(output),
        num_sections=output.count("## "),
        source_citations=output.count("[来源:") + output.count("### "),
    )

    return output, metrics


def run_demo():
    console.print(Panel(
        "[bold]CrewAI 方案：角色协作范式[/bold]\n"
        "研究规划师 → 技术研究员 → 技术分析师 → 报告撰写者\n"
        "组织隐喻: 角色 + 任务链 + 自动上下文传递",
        title="03 CrewAI Solution",
        border_style="blue",
    ))

    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[red]需要设置 DEEPSEEK_API_KEY 环境变量[/red]")
        return

    # 展示团队结构
    table = Table(title="研究团队结构（CrewAI 的组织隐喻）")
    table.add_column("角色", style="cyan")
    table.add_column("Agent", style="green")
    table.add_column("职责")
    table.add_row("研究规划师", "planner", "拆解主题 → 子问题列表")
    table.add_row("技术研究员", "researcher", "逐个研究子问题 → 分析摘要")
    table.add_row("技术分析师", "analyst", "综合分析 → 关键洞察")
    table.add_row("报告撰写者", "writer", "生成结构化报告")
    console.print(table)

    console.print("\n[bold yellow]开始执行研究任务...[/bold yellow]")
    report, metrics = run_benchmark()

    console.print(Panel(report[:800], title="研究报告", border_style="green"))
    console.print(metrics.to_table())

    console.print(Panel(
        "[bold]CrewAI 方案的架构特点[/bold]\n\n"
        "代码组织:\n"
        "  - 4 个 Agent 定义: 每个是 Role + Goal + Backstory\n"
        "  - 4 个 Task 定义: 描述 + 期望输出 + 分配 Agent\n"
        "  - 1 个 Crew 组装: Sequential 流程\n\n"
        "设计优势:\n"
        "  1. 代码量最小（~60 行核心逻辑）\n"
        "  2. 角色语义清晰，非技术人员也能理解\n"
        "  3. 自动上下文传递，不需要定义 State\n"
        "  4. 快速原型: 改角色定义就能改行为\n\n"
        "设计代价:\n"
        "  1. 控制流不透明（你不知道 Researcher 什么时候调用 LLM）\n"
        "  2. 无法精确控制执行路径（不能加条件分支）\n"
        "  3. Token 统计不可用（框架隐藏了底层调用）\n"
        "  4. 调试困难（出问题时不知道是哪一步出错）\n\n"
        "关键 Tradeoff: 简洁性 vs 控制力",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
