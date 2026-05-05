"""
04_claude_sdk_solution.py — 基准任务 Claude SDK 实现：自主循环范式

设计思考：
Claude SDK (anthropic) 处理这个"研究→分析→报告"任务的思路是：
用 Agent 循环 + 手写编排逻辑，精确控制每一步。

流程：
    plan() → [for each sub_topic] → research() → analyze() → write_report()
    手写循环  +  手写状态管理  +  SDK 标准化 API 调用

与 LangGraph 和 CrewAI 方案的核心差异：
- LangGraph: 声明式图编排（定义结构，框架执行）
- CrewAI: 声明式角色定义（定义角色，框架协调）
- Claude SDK: 命令式手写（自己写每一步）

这看起来"低级"，但正是 Claude SDK 的哲学：
- 不隐藏复杂度，让你完全掌控
- 每一步的决策（何时调用 LLM、何时搜索、何时分析）都是显式的
- 适合需要精确控制的场景，不适合快速原型

运行方式：
    cp ../03-claude-agent-sdk/.env.example .env  # 填入 ANTHROPIC_API_KEY
    pip install anthropic python-dotenv rich
    python 04_claude_sdk_solution.py
"""

from __future__ import annotations

import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from anthropic import Anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_task_def = importlib.import_module("01_task_definition")
BenchmarkTask = _task_def.BenchmarkTask
EvaluationMetrics = _task_def.EvaluationMetrics
search_knowledge = _task_def.search_knowledge

load_dotenv()
console = Console()

ANTHROPIC_MODEL = "claude-sonnet-4-6"


# ── 1. 工具定义（Claude SDK 格式）──────────────────────────────
# Claude SDK 的工具用 JSON Schema 定义
# 对比 Phase 1: 手写 ToolRegistry
# 对比 LangGraph: @tool 装饰器自动生成
# 对比 CrewAI: crewai-tools 内置工具

SEARCH_TOOL = {
    "name": "search_knowledge",
    "description": "搜索内置知识库获取技术信息。返回相关条目的内容摘要。",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或问题",
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
            },
        },
        "required": ["query"],
    },
}

TOOLS = [SEARCH_TOOL]


# ── 2. Claude 客户端封装 ───────────────────────────────────────

class ResearchClient:
    """Claude SDK 的简单封装"""

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def send(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
    ):
        kwargs: dict = {"model": ANTHROPIC_MODEL, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return self.client.messages.create(**kwargs)

    @staticmethod
    def get_text(response) -> str:
        """从响应中提取文本"""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    @staticmethod
    def get_tool_uses(response) -> list:
        """从响应中提取工具调用"""
        return [b for b in response.content if b.type == "tool_use"]


# ── 3. Agent 循环 — 手写但精确 ─────────────────────────────────
# 这就是 Claude SDK 的"自主循环"：
# 你写每一步，每一步都是明确的决策
# 对比 LangGraph: 同样的流程用图结构声明
# 对比 CrewAI: 同样的流程用角色定义

class ResearchAgent:
    """手写研究 Agent 循环

    没有 StateGraph，没有 Agent 角色定义。
    就是纯 Python 类 + 方法调用。
    这是最"原始"但也是最灵活的方案。
    """

    def __init__(self, client: ResearchClient):
        self.client = client
        self.metrics = EvaluationMetrics()
        self.llm_calls = 0
        self.search_calls = 0

    def run(self, task: BenchmarkTask) -> str:
        t0 = time.perf_counter()

        # 阶段 1: 规划
        plan = self._plan(task)
        if not plan:
            plan = task.sub_topics

        console.print(Panel(
            "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan)),
            title="研究计划（Claude SDK 生成）",
            border_style="cyan",
        ))

        # 阶段 2: 逐题研究
        findings = []
        for i, sub_topic in enumerate(plan):
            console.print(f"\n  [green][研究 {i+1}/{len(plan)}] {sub_topic[:60]}...[/green]")
            finding = self._research(sub_topic)
            findings.append(finding)
            console.print(f"  [dim]{finding[:120]}...[/dim]")

        # 阶段 3: 分析
        console.print(f"\n  [yellow][分析] 综合 {len(findings)} 个研究结果...[/yellow]")
        analysis = self._analyze(task, findings)
        console.print(f"  [dim]{analysis[:120]}...[/dim]")

        # 阶段 4: 生成报告
        console.print(f"\n  [magenta][报告] 生成最终报告...[/magenta]")
        report = self._write_report(task, findings, analysis)

        elapsed = time.perf_counter() - t0
        self.metrics.total_latency_seconds = elapsed
        self.metrics.num_llm_calls = self.llm_calls
        self.metrics.num_tool_calls = self.search_calls
        self.metrics.output_length = len(report)
        self.metrics.num_sections = report.count("## ")
        self.metrics.source_citations = report.count("[来源:")

        return report

    def _plan(self, task: BenchmarkTask) -> list[str]:
        """规划阶段"""
        response = self.client.send(
            messages=[{"role": "user", "content": f"研究主题: {task.title}\n\n{task.topic}"}],
            system="你是研究规划专家。将主题拆解为 4-6 个具体子问题。每行一个，不要编号。",
            max_tokens=512,
        )
        self.llm_calls += 1
        text = self.client.get_text(response)
        return [s.strip() for s in text.strip().split("\n") if s.strip()][:6]

    def _research(self, topic: str) -> str:
        """研究阶段：搜索 + 分析"""
        # 先用工具搜索
        search_results = search_knowledge(topic)
        self.search_calls += 1

        context = "\n\n".join(
            f"[来源: {r['topic']}]\n{r['content']}" for r in search_results
        ) if search_results else "未找到相关资料。"

        # 再用 LLM 分析
        response = self.client.send(
            messages=[{"role": "user", "content": f"问题: {topic}\n\n参考资料:\n{context}"}],
            system="你是技术研究员。基于参考资料分析问题。输出 150 字以内的分析摘要。",
            max_tokens=512,
        )
        self.llm_calls += 1
        return self.client.get_text(response)

    def _analyze(self, task: BenchmarkTask, findings: list[str]) -> str:
        """分析阶段"""
        combined = "\n\n---\n\n".join(
            f"子问题 {i+1}: {f}" for i, f in enumerate(findings)
        )
        response = self.client.send(
            messages=[{"role": "user", "content": f"主题: {task.title}\n\n研究结果:\n{combined}"}],
            system="你是技术分析师。提炼 3-5 个关键洞察。每个用一句话概括 + 50字解释。",
            max_tokens=1024,
        )
        self.llm_calls += 1
        return self.client.get_text(response)

    def _write_report(self, task: BenchmarkTask, findings: list[str], analysis: str) -> str:
        """报告撰写阶段"""
        findings_text = "\n\n".join(
            f"### 子问题 {i+1}\n{f}" for i, f in enumerate(findings)
        )
        response = self.client.send(
            messages=[{"role": "user", "content": (
                f"主题: {task.title}\n\n"
                f"研究结果:\n{findings_text}\n\n"
                f"综合分析:\n{analysis}\n\n"
                f"报告格式要求:\n{task.output_format}"
            )}],
            system="你是技术报告撰写专家。用 Markdown 格式输出完整报告。",
            max_tokens=2048,
        )
        self.llm_calls += 1
        return self.client.get_text(response)


# ── 4. 运行 ─────────────────────────────────────────────────────

def run_benchmark() -> tuple[str, EvaluationMetrics]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "需要 ANTHROPIC_API_KEY", EvaluationMetrics()

    task = BenchmarkTask()
    client = ResearchClient(api_key)
    agent = ResearchAgent(client)
    report = agent.run(task)
    return report, agent.metrics


def run_demo():
    console.print(Panel(
        "[bold]Claude SDK 方案：自主循环范式[/bold]\n"
        "plan() → for each → research() → analyze() → write_report()\n"
        "纯 Python 类 + 方法调用 = 完全控制",
        title="04 Claude SDK Solution",
        border_style="blue",
    ))

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[yellow]需要设置 ANTHROPIC_API_KEY 环境变量[/yellow]")
        console.print("[dim]Claude SDK 方案使用 Anthropic API，与 LangGraph/CrewAI 方案使用不同的模型[/dim]")
        console.print("[dim]这是设计对比的一部分：三个框架各自使用最优的 LLM 后端[/dim]")

        # 即使没有 API key，也展示代码结构
        console.print(Panel(
            "[bold]代码结构（手写编排）[/bold]\n\n"
            "class ResearchAgent:\n"
            "    def run(self, task) -> str:\n"
            "        plan = self._plan(task)\n"
            "        for sub_topic in plan:          ← 手写循环\n"
            "            findings.append(self._research(sub_topic))\n"
            "        analysis = self._analyze(findings)   ← 手写分析\n"
            "        report = self._write_report(...)     ← 手写报告\n"
            "        return report\n\n"
            "对比 LangGraph（同样的循环用图表示）:\n"
            "    graph.add_conditional_edges('researcher', after, {...})\n\n"
            "对比 CrewAI（同样的循环用角色定义）:\n"
            "    crew = Crew(agents=[...], tasks=[...], process=sequential)\n\n"
            "Claude SDK 最'原始'但最灵活 ——\n"
            "你想加条件分支就加 if，想加重试就加 loop，不需要学图语法。",
            title="代码对比（即使不运行也能看到差异）",
            border_style="cyan",
        ))
        return

    console.print("\n[bold yellow]开始执行研究任务...[/bold yellow]")
    report, metrics = run_benchmark()

    console.print(Panel(report[:800], title="研究报告", border_style="green"))
    console.print(metrics.to_table())

    console.print(Panel(
        "[bold]Claude SDK 方案的架构特点[/bold]\n\n"
        "代码组织:\n"
        "  - 1 个 ResearchAgent 类: 包含全部逻辑\n"
        "  - 4 个私有方法: _plan / _research / _analyze / _write_report\n"
        "  - 手写的 for 循环和状态管理\n\n"
        "设计优势:\n"
        "  1. 完全控制 —— 每一步都显式，没有魔法\n"
        "  2. 灵活 —— 加条件、加重试都是普通 Python 代码\n"
        "  3. 可调试 —— 任何 IDE 都能 debug 这个方法调用链\n"
        "  4. 零依赖 —— 不需要学框架概念\n\n"
        "设计代价:\n"
        "  1. 代码量最大（需要手写循环和状态管理）\n"
        "  2. 没有可视化 —— 代码结构可以任意复杂\n"
        "  3. 没有检查点 —— 进程崩溃就从头开始\n"
        "  4. 需要自己管理所有边界情况\n\n"
        "关键 Tradeoff: 灵活性 vs 代码量",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
