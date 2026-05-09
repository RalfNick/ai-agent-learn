"""
01_task_definition.py — 基准任务定义：三框架对决的共享基线

设计思考：
大多数框架比较文章只是列出功能表（"支持工具调用 ✓"、"支持人机协作 ✓"）。
这类比较毫无意义 —— 框架不是按 checklist 选的，而是按设计哲学选的。

本 showdown 的方法不同：
1. 定义同一个有实际价值的任务（不简单到没区分度，不复杂到需要外部依赖）
2. 用三种框架分别实现
3. 从代码结构、抽象方式、控制流管理、扩展性四个维度做架构对比

基准任务：研究型报告生成

输入：一个技术主题
流程：
  1. 理解并拆解主题 → 子问题列表
  2. 搜索/检索每个子问题的信息（模拟知识库）
  3. 分析信息，提炼关键观点
  4. 生成结构化研究报告（含摘要、分析、结论）

这个任务选得故意"元"（meta）——
用 AI Agent 框架来分析 AI Agent 框架，产出就是他们正在学习的内容。

选这个任务的三个理由：
1. 足够复杂：需要拆解、检索、分析、写作四个阶段
2. 框架中立：不偏向任何一种范式
3. 产出有价值：生成的报告可以直接用于学习

公平比较的原则：
- 使用相同的 LLM（DeepSeek Chat，都通过 LangChain 的 ChatOpenAI 调用）
- 使用相同的知识库（内置 KNOWLEDGE_BASE）
- 相同的输出格式要求
- 每个方案独立可运行

运行方式：
    python 01_task_definition.py  # 查看任务定义和知识库
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ═══════════════════════════════════════════════════════════════
# 1. 共享知识库 — 所有方案使用相同的数据
# ═══════════════════════════════════════════════════════════════

KNOWLEDGE_BASE: dict[str, str] = {
    "langgraph": (
        "LangGraph 是 LangChain 团队开发的 Agent 编排框架（2024 年发布），核心概念是 StateGraph。"
        "它用有向图表示 Agent 工作流：节点（Node）= 处理函数，边（Edge）= 控制流。"
        "设计哲学：显式控制优于隐式推理。核心特性包括："
        "1) 条件路由（conditional edges）—— LLM 的输出决定走哪条路径"
        "2) 检查点持久化（checkpointer）—— 每个节点执行后自动保存完整状态"
        "3) 人机协作（interrupt）—— 在关键节点暂停，等待人类输入"
        "4) 子图（subgraph）—— 大图可以嵌套小图，支持模块化设计。"
        "优势：精确的控制流、可调试性强、生产级可靠性。"
        "劣势：学习曲线陡峭、对简单任务过于复杂、代码量较大。"
        "适用场景：需要精确控制的企业级 Agent 系统、多步骤带质量检查的管道。"
        "2026 年 PyPI 月下载量 4700 万+，是企业级 Agent 开发的事实标准。"
    ),
    "crewai": (
        "CrewAI 是一个多智能体协作框架（2024 年发布），核心概念是 Agent（角色）、Task（任务）、Crew（团队）。"
        "它用组织隐喻来建模多 Agent 协作：每个 Agent 有角色（Role）、目标（Goal）、背景故事（Backstory）。"
        "设计哲学：定义好谁做什么比定义数据怎么流更直觉。核心特性："
        "1) 序列流程（Sequential）—— A 完成后 B 开始，自动上下文传递"
        "2) 层级流程（Hierarchical）—— Manager 自动分配任务给合适的 Agent"
        "3) 工具集成（crewai-tools）—— 内置搜索、抓取、文件操作工具。"
        "优势：快速原型化、非技术人员也能理解、代码简洁。"
        "劣势：控制流不透明、调试困难、不适合需要条件分支的场景。"
        "适用场景：角色明确的协作场景（如产品分析团队、内容创作流水线）。"
        "2026 年 GitHub 19k+ stars，是最受欢迎的多 Agent 框架之一。"
    ),
    "claude_sdk": (
        "Claude Agent SDK 是 Anthropic 提供的 Agent 开发工具包（2025 年发布），"
        "核心是 MCP（Model Context Protocol）协议和 Agent 循环 API。"
        "设计哲学：Agent 是自主行动者，安全是一等公民。核心特性："
        "1) 内置工具——文件读写、Shell 执行、网络请求开箱即用"
        "2) Guardrails——输入验证和输出过滤内建于 Agent 定义"
        "3) Handoff——Agent 间控制权与上下文的安全转移"
        "4) MCP 集成——通过标准协议连接 1000+ 外部工具。"
        "优势：安全设计、与 Anthropic 生态深度集成、MCP 协议标准化。"
        "劣势：编排能力弱于 LangGraph、多 Agent 支持不如 CrewAI、锁定 Anthropic 生态。"
        "适用场景：自主代码操作、安全敏感场景、需要 MCP 工具生态的项目。"
        "Claude SDK 定位在工具层而非编排层，与 LangGraph 互补而非竞争。"
    ),
    "agent_architecture": (
        "AI Agent 的六层架构模型包括："
        "1) 执行引擎层——Agent 循环（ReAct、Plan-Execute、Reflection）"
        "2) 工具系统层——工具注册、schema 生成、执行沙箱化"
        "3) Prompt 引擎层——系统提示、消息管理、token 预算控制"
        "4) 记忆系统层——短期记忆（对话历史）、长期记忆（向量存储）、工作记忆（scratchpad）"
        "5) 编排层——单 Agent 路由 vs 多 Agent 协作（Supervisor、Swarm、Graph）"
        "6) 安全与监控层——Guardrails、可观测性、审计日志。"
        "理解这六层是掌握 Agent 框架设计的核心。"
        "不同框架在不同层次有不同的抽象方式："
        "LangGraph 在编排层最强（状态图），Claude SDK 在安全层最突出（Guardrails），"
        "CrewAI 在编排层用组织隐喻（角色+流程）替代了图模型。"
    ),
    "framework_selection": (
        "Agent 框架选型的决策框架："
        "1) 控制力需求——需要精确控制执行路径选 LangGraph，快速原型选 CrewAI"
        "2) 团队规模——小团队选 CrewAI（低学习成本），大团队选 LangGraph（可维护性）"
        "3) 安全要求——高安全需求选 Claude SDK（Guardrails 是一等公民）"
        "4) 任务复杂度——简单线性任务用 CrewAI，复杂条件分支用 LangGraph"
        "5) 生态依赖——已有 LangChain 项目选 LangGraph，Anthropic 用户选 Claude SDK"
        "6) 可观测性——LangGraph 的图结构天然支持调试和可视化。"
        "没有银弹：框架选择取决于具体场景的约束条件和优先级。"
        "多数生产系统会混合使用：LangGraph 做编排 + Claude SDK 做工具调用。"
    ),
    "multi_agent_patterns": (
        "多 Agent 系统的三种主流架构模式："
        "1) Supervisor 模式——一个主 Agent 负责任务分解和路由，其他 Agent 执行具体任务"
        "   优势：集中控制、便于追踪。劣势：Supervisor 成为瓶颈。"
        "   LangGraph 实现：Supervisor 节点 + 条件路由，CrewAI 实现：Hierarchical 流程。"
        "2) Swarm 模式——Agent 之间直接通信，没有中心协调者"
        "   优势：高灵活、无单点故障。劣势：协调复杂、可能出现冲突。"
        "   代表框架：OpenAI Swarm。"
        "3) Graph 模式——用有向图定义 Agent 间的控制流和数据流"
        "   优势：显式、可调试、可优化。劣势：需要预先设计图结构。"
        "   LangGraph 天然支持这种模式。"
        "选择哪种模式取决于任务的可分解性和 Agent 间的依赖关系。"
    ),
}


# ═══════════════════════════════════════════════════════════════
# 2. 基准任务定义
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkTask:
    """基准测试任务定义"""

    title: str = "AI Agent 框架选型研究报告"
    topic: str = (
        "分析当前主流的 AI Agent 开发框架（LangGraph、CrewAI、Claude Agent SDK），"
        "包括它们的核心设计哲学、架构特点、适用场景、优劣势对比，"
        "并给出面向企业开发者的选型建议。"
    )
    sub_topics: list[str] = field(default_factory=lambda: [
        "LangGraph 的核心设计理念和技术特点",
        "CrewAI 的多 Agent 协作模式",
        "Claude Agent SDK 的安全设计哲学",
        "多 Agent 系统的三种架构模式（Supervisor/Swarm/Graph）",
        "框架选型的关键维度和决策框架",
    ])
    evaluation_criteria: dict[str, str] = field(default_factory=lambda: {
        "research_completeness": "是否覆盖了所有子主题",
        "analysis_depth": "分析是否有深度，不仅仅是表面描述",
        "structure": "报告结构是否清晰（摘要→分析→对比→建议）",
        "actionability": "结论和建议是否可执行",
        "source_accuracy": "引用的事实是否准确",
    })
    output_format: str = (
        "# {title}\n\n"
        "## 摘要\n"
        "一句话核心结论 + 3-5 个关键发现\n\n"
        "## 各框架分析\n"
        "对每个框架的核心设计、优势、劣势、适用场景的分析\n\n"
        "## 架构对比\n"
        "用表格比较三个框架在关键维度的差异\n\n"
        "## 选型建议\n"
        "面向不同场景的具体建议（小团队/大团队、简单任务/复杂任务、低安全/高安全）\n\n"
        "## 结论\n"
        "最终建议和未来展望"
    )


@dataclass
class EvaluationMetrics:
    """评估指标"""
    total_tokens: int = 0
    total_latency_seconds: float = 0.0
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    output_length: int = 0
    num_sections: int = 0  # 报告包含的章节数
    source_citations: int = 0  # 引用的知识库条目数

    def to_table(self) -> Table:
        table = Table(title="执行指标")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        table.add_row("总 Token 消耗", f"{self.total_tokens:,}")
        table.add_row("总延迟", f"{self.total_latency_seconds:.1f}s")
        table.add_row("LLM 调用次数", str(self.num_llm_calls))
        table.add_row("工具调用次数", str(self.num_tool_calls))
        table.add_row("输出长度", f"{self.output_length:,} 字符")
        table.add_row("报告章节数", str(self.num_sections))
        table.add_row("引用源数", str(self.source_citations))
        return table


def search_knowledge(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """简单的关键词匹配搜索（所有方案使用同一个实现以确保公平）"""
    results = []
    query_words = set(query.lower().split())
    for topic, content in KNOWLEDGE_BASE.items():
        score = sum(1 for w in query_words if w in topic.lower() or w in content.lower())
        if score > 0:
            results.append({"topic": topic, "content": content, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def print_task_definition():
    """打印基准任务定义"""
    task = BenchmarkTask()

    console.print(Panel(
        "[bold]基准任务定义[/bold]\n\n"
        f"主题: {task.title}\n"
        f"任务: {task.topic}\n\n"
        "流程: 拆解主题 → 搜索信息 → 分析提炼 → 生成报告",
        title="01 Task Definition",
        border_style="blue",
    ))

    # 子主题
    sub_table = Table(title="研究子主题")
    sub_table.add_column("#", style="cyan")
    sub_table.add_column("子主题", style="yellow")
    for i, sub in enumerate(task.sub_topics, 1):
        sub_table.add_row(str(i), sub)
    console.print(sub_table)

    # 评估标准
    eval_table = Table(title="评估标准")
    eval_table.add_column("维度", style="cyan")
    eval_table.add_column("标准", style="green")
    for dim, std in task.evaluation_criteria.items():
        eval_table.add_row(dim, std)
    console.print(eval_table)

    # 知识库
    kb_table = Table(title="共享知识库（所有方案使用相同数据）")
    kb_table.add_column("条目", style="cyan")
    kb_table.add_column("内容摘要", style="dim")
    for topic, content in KNOWLEDGE_BASE.items():
        kb_table.add_row(topic, content[:120] + "...")
    console.print(kb_table)

    console.print(Panel(
        "[bold]公平比较原则[/bold]\n\n"
        "1. 相同 LLM: 所有方案使用 DeepSeek Chat（通过 ChatOpenAI 调用）\n"
        "2. 相同知识库: 所有方案使用 KNOWLEDGE_BASE（本文件定义）\n"
        "3. 相同任务: 所有方案执行相同的 BenchmarkTask\n"
        "4. 独立运行: 每个方案一个文件，python <file>.py 即可运行\n\n"
        "比较的维度不是「谁更快」「谁更省 token」——\n"
        "而是「同样的需求，不同框架如何用不同的抽象来建模」。\n"
        "架构上的差异才是框架对比的核心价值。",
        title="实验设计",
        border_style="green",
    ))


if __name__ == "__main__":
    print_task_definition()
