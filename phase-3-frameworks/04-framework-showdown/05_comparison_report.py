"""
05_comparison_report.py — 三框架终极对比：架构分析 + 代码并排 + 选型决策框架

设计思考：
前面三个方案（02/03/04）用三种框架实现了同一个任务。
本文件不运行它们（API key 不同），而是做架构层面的深入对比。

大多数框架对比是无效的，因为它们：
- 比较功能 checklist（"支持 X ✓"），而不是架构哲学
- 用玩具示例，看不出框架差异
- 没有回答最关键的问题："我应该用哪个？"

本文件的对比聚焦三个核心维度：
1. 同一任务，三种代码结构的并排对比
2. 抽象层分析：每个框架隐藏了什么？暴露了什么？
3. 决策框架：基于场景特征匹配框架

"不是选哪个框架，而是理解为什么"
—— 理解每个框架在什么场景下是最优选择。

运行方式：
    python 05_comparison_report.py
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree

console = Console()


# ═══════════════════════════════════════════════════════════════
# 1. 代码并排对比
# ═══════════════════════════════════════════════════════════════

LANGGRAPH_PLANNER = '''# LangGraph: 图编排（声明式状态 + 节点 + 边）
class ResearchState(TypedDict):
    sub_topics: list[str]
    current_index: int
    research_results: list[str]

def planner_node(state: ResearchState) -> dict:
    """节点函数：State → partial State"""
    response = llm.invoke(...)
    return {"sub_topics": parse(response)}

def after_researcher(state: ResearchState) -> str:
    """路由函数：显式的条件逻辑"""
    if state["current_index"] < len(state["sub_topics"]):
        return "researcher"
    return "analyzer"

# 声明式组装
graph = StateGraph(ResearchState)
graph.add_node("planner", planner_node)
graph.add_conditional_edges("researcher", after_researcher, {...})
app = graph.compile()
result = app.invoke(initial_state)'''

CREWAI_PLANNER = '''# CrewAI: 角色协作（角色定义 + 任务链 + 自动上下文）
planner = Agent(
    role="研究规划师",
    goal="将研究主题拆解为子问题",
    backstory="你是一位资深研究规划师...",
    llm=llm,
)

plan_task = Task(
    description="研究主题: {topic}...",
    expected_output="4-6 个子问题列表",
    agent=planner,
)

# 声明式组装
crew = Crew(
    agents=[planner, researcher, analyst, writer],
    tasks=[plan_task, research_task, ...],
    process=Process.sequential,
)
result = crew.kickoff(inputs={"topic": "..."})'''

CLAUDE_SDK_PLANNER = '''# Claude SDK: 自主循环（手写类 + 方法调用 + 完全控制）
class ResearchAgent:
    def run(self, task: BenchmarkTask) -> str:
        # 阶段 1: 手写规划调用
        plan = self._plan(task)

        # 阶段 2: 手写研究循环
        findings = []
        for i, sub_topic in enumerate(plan):
            finding = self._research(sub_topic)
            findings.append(finding)

        # 阶段 3: 手写分析调用
        analysis = self._analyze(task, findings)

        # 阶段 4: 手写报告生成
        return self._write_report(task, findings, analysis)

    def _plan(self, task) -> list[str]:
        response = self.client.send(
            messages=[{"role": "user", "content": ...}],
            system="你是研究规划专家...",
        )
        self.llm_calls += 1
        return parse(response)

agent = ResearchAgent(client)
result = agent.run(task)'''


def print_code_comparison():
    """同任务、三框架、代码并排"""
    console.print(Panel(
        "[bold]同一任务，三种代码结构[/bold]\n"
        "都实现了 规划 → 研究 × N → 分析 → 报告 这个工作流\n"
        "但代码结构和抽象方式截然不同",
        title="代码并排对比",
        border_style="blue",
    ))

    console.print(Syntax(LANGGRAPH_PLANNER, "python", line_numbers=False))
    console.print(Syntax(CREWAI_PLANNER, "python", line_numbers=False))
    console.print(Syntax(CLAUDE_SDK_PLANNER, "python", line_numbers=False))


# ═══════════════════════════════════════════════════════════════
# 2. 架构维度对比
# ═══════════════════════════════════════════════════════════════

def print_architecture_comparison():
    """多维度架构对比"""
    console.print(Panel(
        "[bold]多维度架构对比[/bold]",
        title="架构分析",
        border_style="blue",
    ))

    dims = Table(title="核心维度对比")
    dims.add_column("维度", style="cyan", width=18)
    dims.add_column("LangGraph", style="green", width=30)
    dims.add_column("CrewAI", style="yellow", width=30)
    dims.add_column("Claude SDK", style="magenta", width=30)

    dims.add_row(
        "抽象隐喻",
        "有向图（StateGraph）\n节点=处理步骤\n边=控制流",
        "组织（Organizational）\nAgent=角色\nCrew=团队",
        "自主代理（Agentic）\nAgent=自主循环\n工具=能力",
    )
    dims.add_row(
        "状态管理",
        "显式 TypedDict\n所有节点共享\n自动持久化",
        "隐式自动传递\n前 Task 输出 → 后 Task 输入\n开发者不可见",
        "手动管理\nPython 变量\n无内置持久化",
    )
    dims.add_row(
        "控制流",
        "声明式图\n条件边路由\n可视化",
        "声明式流程\nSequential/Hierarchical\n不透明",
        "命令式代码\nif/for/while\n完全控制",
    )
    dims.add_row(
        "扩展方式",
        "加节点+加边\n子图（subgraph）\n模块化",
        "加 Agent+加 Task\n角色组合\n平铺式",
        "加方法+加类\nOOP 继承/组合\n任意组织",
    )
    dims.add_row(
        "学习曲线",
        "陡峭\n需要学图语法+概念\n但值得",
        "平缓\n概念直觉（角色/任务）\n快速上手",
        "平缓（作为 API）\n只需会 Python\n陡峭（做编排）",
    )
    dims.add_row(
        "调试体验",
        "优秀\n图可视化+时间旅行\ncheckpoint 回放",
        "一般\n只知道哪个 Agent 失败\n不知道为什么",
        "标准 Python\ndebugger 可断点\n没有特殊工具",
    )
    dims.add_row(
        "安全模型",
        "流程级\ninterrupt 暂停\n人工审批",
        "角色级\nAgent 能力边界\n无 Guardrails",
        "定义级\nGuardrails 是 Agent 一部分\n输入/输出过滤",
    )

    console.print(dims)


# ═══════════════════════════════════════════════════════════════
# 3. 抽象层分析
# ═══════════════════════════════════════════════════════════════

def print_abstraction_analysis():
    """每个框架隐藏和暴露了什么"""
    console.print(Panel(
        "[bold]抽象层分析：每个框架隐藏了什么？暴露了什么？[/bold]\n"
        "框架的本质是抽象——好的抽象让你专注于解决业务问题，\n"
        "但不好的抽象会在你遇到边界情况时成为障碍。",
        title="抽象层分析",
        border_style="blue",
    ))

    abs_table = Table(title="隐藏 vs 暴露")
    abs_table.add_column("框架", style="cyan", width=12)
    abs_table.add_column("隐藏了什么（你不需要管）", style="green", width=40)
    abs_table.add_column("暴露了什么（你需要管的）", style="yellow", width=40)
    abs_table.add_column("逃生舱（遇到极限怎么办）", style="red", width=35)

    abs_table.add_row(
        "LangGraph",
        "- 执行循环\n- 状态持久化\n- 检查点保存\n- 消息格式转换",
        "- 状态 schema 定义\n- 节点函数实现\n- 路由逻辑\n- 图结构设计",
        "自定义节点可以是任意 Python 代码\n可以绕过 StateGraph 直接调用 LLM\n可以用 subgraph 局部降级",
    )
    abs_table.add_row(
        "CrewAI",
        "- 上下文传递\n- Agent 协调\n- Task 调度\n- 消息管理",
        "- 角色定义\n- Task 描述\n- 流程选择\n- 工具集成",
        "Agent 可以调用任意 Python 函数\n可以用 allow_delegation 控制\n但无法改变框架的调度逻辑",
    )
    abs_table.add_row(
        "Claude SDK",
        "- API 格式\n- 消息编码\n- 流式传输\n- 工具 schema 标准化",
        "- 循环逻辑\n- 状态管理\n- 错误处理\n- 一切编排",
        "本身就是最底层\n可以在上层加任何模式\n可以完全替换任何部分",
    )

    console.print(abs_table)


# ═══════════════════════════════════════════════════════════════
# 4. 选型决策框架
# ═══════════════════════════════════════════════════════════════

def print_decision_framework():
    """企业选型决策框架"""
    console.print(Panel(
        "[bold]选型决策框架：不是选哪个，而是理解为什么[/bold]",
        title="决策框架",
        border_style="blue",
    ))

    # 场景-推荐矩阵
    matrix = Table(title="场景 → 推荐矩阵")
    matrix.add_column("场景特征", style="cyan", width=25)
    matrix.add_column("推荐框架", style="green", width=15)
    matrix.add_column("理由", width=55)

    matrix.add_row(
        "小团队、快速验证想法",
        "CrewAI",
        "学习成本最低，角色定义直觉，几行代码就能跑",
    )
    matrix.add_row(
        "大团队、生产级系统",
        "LangGraph",
        "显式状态+图结构保证可维护性，checkpoint 保可靠性",
    )
    matrix.add_row(
        "需要精确控制执行路径",
        "LangGraph",
        "条件边+路由函数 = 声明式的精确控制",
    )
    matrix.add_row(
        "角色明确的协作流程",
        "CrewAI",
        "组织隐喻天然适合多角色场景，代码最简洁",
    )
    matrix.add_row(
        "高安全要求",
        "Claude SDK / LangGraph",
        "Claude SDK 的 Guardrails + LangGraph 的 interrupt 互补",
    )
    matrix.add_row(
        "需要与文件系统交互",
        "Claude SDK / LangGraph",
        "Claude SDK 内置文件工具，LangGraph 需自定义工具",
    )
    matrix.add_row(
        "已有 LangChain 项目",
        "LangGraph",
        "共享 LangChain 生态，LangGraph 本身就是 LangChain 家族",
    )
    matrix.add_row(
        "已有 Anthropic 生态项目",
        "Claude SDK + LangGraph",
        "Claude SDK 做工具层 + LangGraph 做编排层（最强组合）",
    )

    console.print(matrix)

    # 组合使用建议
    console.print(Panel(
        "[bold]实际生产建议：框架组合[/bold]\n\n"
        "多数生产系统不会只用一种框架，而是组合：\n\n"
        "简单项目（1-2 个 Agent，线性流程）:\n"
        "  CrewAI 或 手写循环 ← 够用且简单\n\n"
        "中型项目（多 Agent，有条件分支）:\n"
        "  LangGraph 做编排 + LangChain/LlamaIndex 做组件\n\n"
        "大型项目（企业级，有安全合规要求）:\n"
        "  LangGraph 做编排层（控制流、状态管理、checkpointing）\n"
        "  + Claude SDK / 直接 API 做工具层（工具调用、Guardrails）\n"
        "  + MCP 做工具生态（1000+ 标准工具）\n\n"
        "核心原则：\n"
        "  编排层选 LangGraph（当需要精确控制流程时）\n"
        "  工具层选 Claude SDK 或直接 API（当需要安全防护时）\n"
        "  快速原型选 CrewAI（当时间是主要约束时）",
        title="框架组合建议",
        border_style="green",
    ))


# ═══════════════════════════════════════════════════════════════
# 5. 核心洞察
# ═══════════════════════════════════════════════════════════════

def print_insights():
    """对比的核心洞察"""
    insights_tree = Tree("[bold]三框架对比的核心洞察[/bold]")

    insight1 = insights_tree.add(
        "[cyan]洞察 1: 框架分层，而非替代[/cyan]"
    )
    insight1.add(
        "LangGraph 是编排框架（orchestration layer）—— 管理 Agent 间的控制流\n"
        "Claude SDK 是工具框架（tooling layer）—— 提供 LLM 调用和工具执行\n"
        "CrewAI 是协作框架（collaboration layer）—— 用角色隐喻封装多 Agent 协调\n\n"
        "这三者各自解决同一问题的不同层次。它们不是竞品，是可以组合的。"
    )

    insight2 = insights_tree.add(
        "[cyan]洞察 2: 抽象 ≠ 隐藏——抽象是选择性地隐藏[/cyan]"
    )
    insight2.add(
        "LangGraph 隐藏了循环逻辑但暴露了图结构——你需要学图语法\n"
        "CrewAI 隐藏了调度逻辑但暴露了角色定义——你需要写好 Backstory\n"
        "Claude SDK 只隐藏 API 格式——你需要自己管理一切\n\n"
        "好的抽象让你在 80% 的场景里更快，在 20% 的场景里不被阻碍。\n"
        "每个框架的'逃生舱'（abs_table 第三列）决定了它的天花板。"
    )

    insight3 = insights_tree.add(
        "[cyan]洞察 3: 控制链 = 复杂度的来源[/cyan]"
    )
    insight3.add(
        "CrewAI: 你放弃控制 → 复杂度最低 → 但框架出错时你束手无策\n"
        "LangGraph: 你声明控制 → 复杂度适中 → 图结构让你看到全貌\n"
        "Claude SDK: 你完全控制 → 复杂度最高 → 但任何问题你都能修\n\n"
        "没有免费的午餐。选择多少控制 = 选择多少复杂度。"
    )

    insight4 = insights_tree.add(
        "[cyan]洞察 4: 代码结构反映了世界观[/cyan]"
    )
    insight4.add(
        "LangGraph 的世界观: 世界是一个状态机，变化是节点间的转换\n"
        "CrewAI 的世界观: 世界是一个组织，每个人有角色、有目标\n"
        "Claude SDK 的世界观: 世界是可操作的，Agent 是你的工具\n\n"
        "选择框架不只是选择技术，也是选择看待问题的方式。"
    )

    insight5 = insights_tree.add(
        "[cyan]洞察 5: 未来趋势是分层组合[/cyan]"
    )
    insight5.add(
        "MCP（Model Context Protocol）正在成为工具层的标准\n"
        "LangGraph 在编排层占据主流\n"
        "CrewAI 在快速原型场景是首选\n\n"
        "未来的企业级 Agent 架构很可能是：\n"
        "  MCP Server（工具） + LangGraph（编排） + Guardrails（安全）\n"
        "这不是三选一，而是各取所长。"
    )

    console.print(insights_tree)


# ═══════════════════════════════════════════════════════════════
# 6. 运行
# ═══════════════════════════════════════════════════════════════

def run_demo():
    console.print(Panel(
        "[bold]Agent 框架终极对比[/bold]\n"
        "同一任务、三个框架、五个分析维度\n"
        "不是选哪个，而是理解为什么",
        title="05 Comparison Report",
        border_style="blue",
    ))

    print_code_comparison()
    print_architecture_comparison()
    print_abstraction_analysis()
    print_decision_framework()
    print_insights()

    console.print(Panel(
        "[bold]关于为什么没有运行结果[/bold]\n\n"
        "三个方案的 API 要求和 key 不同：\n"
        "- 02_langgraph_solution.py: DeepSeek API\n"
        "- 03_crewai_solution.py: DeepSeek API\n"
        "- 04_claude_sdk_solution.py: Anthropic API\n\n"
        "它们各自独立可运行（python 0X_xxx.py）。\n"
        "本文件聚焦架构层面的对比——\n"
        "代码结构、抽象设计、决策框架——\n"
        "这些比运行数字更有价值。\n\n"
        "如果需要量化对比，可以统一 API 后运行三个方案。",
        title="说明",
        border_style="dim",
    ))


if __name__ == "__main__":
    run_demo()
