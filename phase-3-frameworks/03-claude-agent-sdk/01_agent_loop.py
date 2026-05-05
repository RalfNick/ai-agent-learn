"""
01_agent_loop.py — Claude SDK Agent 循环：从手写 while 到 SDK 封装的三种范式

设计思考：
Phase 1 的 ExecutionEngine 手写了完整的 Agent 循环：
    while not done: think → act → observe → repeat

这个循环看似简单，实则隐藏了五个关键设计决策：
1. 终止条件：什么时候停止？（最终答案 vs 最大步数 vs 置信度阈值）
2. 状态传递：消息历史如何累积？（追加 vs 压缩 vs 滑动窗口）
3. 工具执行：同步还是异步？失败怎么办？
4. 流式输出：用户如何看到 Agent 的思考过程？
5. 结构化输出：如何保证 Agent 返回可解析的数据？

Claude SDK (anthropic) 的定位是"工具层"——它标准化了 API 调用的格式，但循环逻辑
仍然由开发者掌控。这与 LangGraph（编排层封装）和 CrewAI（组织层封装）形成对比。

本文件实现三种 Agent 循环范式，每种揭示不同的设计权衡：

  SimpleAgent    — 基础 ReAct 循环，对应 Phase 1 的 ExecutionEngine
  PlanExecuteAgent — 先规划后执行，对应 LangGraph 的 Plan-and-Execute
  ReflectionAgent — 执行后自我反思，加入质量检查循环

运行方式：
    cp .env.example .env  # 填入 ANTHROPIC_API_KEY
    pip install -r requirements.txt
    python 01_agent_loop.py
"""

from __future__ import annotations

import ast
import json
import operator
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import Message, ContentBlock, ToolUseBlock, TextBlock
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.live import Live
from rich.text import Text

load_dotenv()
console = Console()


# ═══════════════════════════════════════════════════════════════
# 1. 基础设施：工具系统与 SDK 初始化
# ═══════════════════════════════════════════════════════════════

@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    input_args: dict[str, Any]
    output: str
    elapsed_ms: float
    success: bool
    error: str | None = None


@dataclass
class AgentStep:
    """Agent 执行步骤的记录（对应 Phase 1 的 Step）"""
    step_num: int
    thinking: str = ""
    tool_results: list[ToolResult] = field(default_factory=list)
    final_answer: str = ""
    elapsed_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)


# ── 工具定义（Claude SDK 的 JSON Schema 格式）────────────────
# 与 Phase 1 的 ToolRegistry 不同的是，这里的 schema 是完全手写的 dict。
# OpenAI/Anthropic 都采用这种格式，实现了跨平台兼容。
# 缺点：没有类型安全，没有自动 schema 生成。

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "calculate",
        "description": "计算数学表达式。支持加减乘除、幂运算、三角函数。如果表达式包含变量，请先用 search_knowledge 查询公式。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，支持 +, -, *, /, **, sin, cos, sqrt",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "搜索内置知识库获取事实信息。返回相关条目的内容摘要。",
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
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "analyze_data",
        "description": "对提供的数据列表进行统计分析，返回：平均值、中位数、最大值、最小值、标准差。",
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "要分析的数字列表",
                },
                "operation": {
                    "type": "string",
                    "enum": ["summary", "trend", "outliers"],
                    "description": "分析类型：summary=统计摘要, trend=趋势, outliers=异常检测",
                },
            },
            "required": ["data"],
        },
    },
]

# 模拟知识库
KNOWLEDGE_DB = {
    "langgraph": "LangGraph 是 LangChain 团队开发的 Agent 编排框架。核心是 StateGraph（有向图），"
                 "节点是处理函数，边是控制流。支持条件路由、检查点持久化、人机协作（interrupt）。"
                 "2026 年 PyPI 月下载量 4700 万+。适用于需要精确控制执行流程的生产 Agent。",
    "crewai": "CrewAI 是多 Agent 协作框架。核心抽象：Agent（角色+目标+背景）、Task、Crew（团队）。"
              "支持 Sequential 和 Hierarchical 两种流程。优势是快速原型化多角色工作流。"
              "2026 年 GitHub 19k+ stars。适用于角色明确的协作场景。",
    "agent_loop": "Agent 循环是 AI Agent 的核心执行模式。基本形态：观察→思考→行动→观察 的 ReAct 循环。"
                  "变体包括：Plan-and-Execute（先规划后执行）、Reflection（执行后自我反思）、"
                  "LLM Compiler（并行执行）。循环设计的关键决策：终止条件、状态管理、错误恢复。",
    "react": "ReAct（Reasoning + Acting）是 2022 年提出的 Agent 模式。核心思想：LLM 交替进行"
             "推理（Thought）和行动（Action），行动结果（Observation）反馈给下一步推理。"
             "优势是透明可解释，劣势是线性执行效率低。",
    "ai_agent": "AI Agent 是能够自主感知环境、做出决策、执行行动的智能系统。"
                "四要素：感知（输入）、推理（思考）、行动（工具调用）、记忆（状态）。"
                "2026 年主流框架包括 LangGraph、CrewAI、AutoGen、Agno 等。",
}

ANTHROPIC_MODEL = "claude-sonnet-4-6"


# ── 工具实现函数 ──────────────────────────────────────────────

def _calculate(expression: str) -> str:
    """安全的数学表达式求值（使用 AST 而非 eval）"""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
    }
    import math as _math

    _allowed = {
        "sin": _math.sin, "cos": _math.cos, "sqrt": _math.sqrt,
        "pi": _math.pi, "e": _math.e, "abs": abs, "round": round,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _allowed:
                args = [_eval(a) for a in node.args]
                return _allowed[name](*args)
            raise ValueError(f"不允许的函数: {name}")
        if isinstance(node, ast.Name):
            if node.id in _allowed:
                return _allowed[node.id]
            raise ValueError(f"未知变量: {node.id}")
        raise ValueError(f"不支持的语法: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        return f"{expression} = {result}"
    except Exception as e:
        return f"表达式错误: {e}"


def _search_knowledge(query: str, top_k: int = 3) -> str:
    """简单的关键词匹配搜索"""
    results = []
    query_words = set(query.lower().split())
    for topic, content in KNOWLEDGE_DB.items():
        score = sum(1 for w in query_words if w in topic.lower() or w in content.lower())
        if score > 0:
            results.append((score, topic, content))
    results.sort(key=lambda x: x[0], reverse=True)
    if not results:
        return "未找到相关知识。"
    return "\n---\n".join(f"[{topic}] {content[:300]}" for _, topic, content in results[:top_k])


def _analyze_data(data: list[float], operation: str = "summary") -> str:
    """统计分析"""
    n = len(data)
    if n == 0:
        return "数据为空。"
    mean = sum(data) / n
    sorted_data = sorted(data)
    median = sorted_data[n // 2] if n % 2 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5

    if operation == "summary":
        return (
            f"统计摘要 (n={n}): 均值={mean:.2f}, 中位数={median:.2f}, "
            f"最小值={sorted_data[0]:.2f}, 最大值={sorted_data[-1]:.2f}, 标准差={std_dev:.2f}"
        )
    elif operation == "trend":
        if n < 2:
            return "数据点不足，无法分析趋势。"
        changes = [data[i] - data[i - 1] for i in range(1, n)]
        avg_change = sum(changes) / len(changes)
        direction = "上升" if avg_change > 0 else "下降" if avg_change < 0 else "平稳"
        return f"趋势: {direction}（平均变化 {avg_change:+.2f}/步），起始={data[0]:.2f}, 结束={data[-1]:.2f}"
    elif operation == "outliers":
        q1 = sorted_data[n // 4]
        q3 = sorted_data[3 * n // 4]
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = [x for x in data if x < lower or x > upper]
        return f"异常检测 (IQR): 下界={lower:.2f}, 上界={upper:.2f}, 异常值={outliers if outliers else '无'}"


TOOL_EXECUTORS: dict[str, Callable[..., str]] = {
    "calculate": lambda **kw: _calculate(kw["expression"]),
    "search_knowledge": lambda **kw: _search_knowledge(kw["query"], kw.get("top_k", 3)),
    "analyze_data": lambda **kw: _analyze_data(kw["data"], kw.get("operation", "summary")),
}


# ═══════════════════════════════════════════════════════════════
# 2. Claude SDK 客户端封装
# ═══════════════════════════════════════════════════════════════

class ClaudeClient:
    """Claude SDK 的轻量封装，统一处理消息格式转换

    Claude API 的消息格式与 OpenAI 不同：
    - OpenAI: messages = [{"role": "user", "content": "..."}]
    - Claude: messages 也类似，但 content 可以是 text 或 ContentBlock 列表
    - Claude 的 tool_use 是 ContentBlock，tool_result 是 user role 的特殊 block

    这个封装类处理这些兼容性细节，让上层 Agent 逻辑可以专注于循环设计。
    """

    def __init__(self, api_key: str | None = None, model: str = ANTHROPIC_MODEL):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def send(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> Message:
        """发送消息到 Claude，返回原始 Message 对象"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        return self.client.messages.create(**kwargs)

    @staticmethod
    def extract_text_and_tools(response: Message) -> tuple[str, list[ToolUseBlock]]:
        """从响应中提取文本内容和工具调用"""
        texts: list[str] = []
        tool_uses: list[ToolUseBlock] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)
        return "\n".join(texts), tool_uses

    @staticmethod
    def make_assistant_message(response: Message) -> dict[str, Any]:
        """构建 assistant 角色的消息"""
        content: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return {"role": "assistant", "content": content}

    @staticmethod
    def make_tool_result_message(tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        """构建工具结果消息"""
        content = [
            {
                "type": "tool_result",
                "tool_use_id": r["tool_use_id"],
                "content": r["content"],
            }
            for r in tool_results
        ]
        return {"role": "user", "content": content}

    def execute_tools(self, tool_uses: list[ToolUseBlock]) -> list[dict[str, Any]]:
        """执行工具调用，返回工具结果列表"""
        results = []
        for tu in tool_uses:
            fn = TOOL_EXECUTORS.get(tu.name)
            if fn:
                t0 = time.perf_counter()
                try:
                    output = fn(**tu.input)
                    success = True
                    error = None
                except Exception as e:
                    output = f"工具执行失败: {e}"
                    success = False
                    error = str(e)
                elapsed = (time.perf_counter() - t0) * 1000
                results.append({
                    "tool_use_id": tu.id,
                    "content": output,
                    "_meta": ToolResult(
                        tool_name=tu.name,
                        input_args=dict(tu.input),
                        output=output,
                        elapsed_ms=elapsed,
                        success=success,
                        error=error,
                    ),
                })
        return results


# ═══════════════════════════════════════════════════════════════
# 3. 三种 Agent 循环范式
# ═══════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """Agent 基类 —— 抽象出 Agent 循环的共性

    Phase 1 的 ExecutionEngine 把所有逻辑塞在一个类里。
    这里用继承分离三种范式，让设计差异更清晰。
    """

    def __init__(self, client: ClaudeClient, max_steps: int = 10):
        self.client = client
        self.max_steps = max_steps
        self.steps: list[AgentStep] = []

    @abstractmethod
    def run(self, task: str) -> str:
        ...

    def print_trace(self):
        """打印完整的执行追踪"""
        table = Table(title=f"{self.__class__.__name__} 执行追踪")
        table.add_column("步骤", style="cyan", width=6)
        table.add_column("思考", style="yellow", width=40)
        table.add_column("工具调用", style="green", width=20)
        table.add_column("耗时", style="dim", width=10)

        for step in self.steps:
            thinking = step.thinking[:80] + "..." if len(step.thinking) > 80 else step.thinking
            tools_str = ", ".join(
                f"{t.tool_name}({t.elapsed_ms:.0f}ms)" for t in step.tool_results
            ) or "-"
            table.add_row(
                str(step.step_num),
                thinking,
                tools_str,
                f"{step.elapsed_ms:.0f}ms",
            )
        console.print(table)


class SimpleAgent(BaseAgent):
    """范式 1：基础 ReAct 循环

    这是最原始的 Agent 循环，直接对应 Phase 1 的 ExecutionEngine：

        while step < max_steps:
            think → 如果有工具调用 → 执行工具 → 返回结果 → repeat
                   → 如果没有 → 返回最终答案 → 结束

    设计要点：
    - 终止条件很简单：LLM 不调用工具了 = 任务完成
    - 没有规划阶段，Agent 边走边看
    - 适合：简单工具调用任务（计算、查询、翻译）
    - 不适合：需要多步推理的复杂任务（Agent 容易迷失方向）
    """

    SYSTEM_PROMPT = (
        "你是一个专业的 AI 助手。对于每个问题：\n"
        "1. 如果需要计算或搜索，使用工具\n"
        "2. 在获得足够信息后，给出简洁的最终答案\n"
        "3. 不要重复工具调用 —— 如果一次调用就能得到答案，不要调用第二次"
    )

    def run(self, task: str) -> str:
        self.steps = []
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

        for i in range(1, self.max_steps + 1):
            step = AgentStep(step_num=i)
            t0 = time.perf_counter()

            response = self.client.send(
                messages=messages,
                system=self.SYSTEM_PROMPT,
                tools=TOOLS_SCHEMA,
            )

            text, tool_uses = self.client.extract_text_and_tools(response)
            step.thinking = text[:200]
            step.token_usage = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }

            messages.append(self.client.make_assistant_message(response))

            if not tool_uses:
                step.final_answer = text
                step.elapsed_ms = (time.perf_counter() - t0) * 1000
                self.steps.append(step)
                return text

            tool_results = self.client.execute_tools(tool_uses)
            step.tool_results = [r["_meta"] for r in tool_results]
            step.elapsed_ms = (time.perf_counter() - t0) * 1000
            self.steps.append(step)

            # 清理 _meta 字段后追加到消息历史
            clean_results = [
                {"tool_use_id": r["tool_use_id"], "content": r["content"]}
                for r in tool_results
            ]
            messages.append(self.client.make_tool_result_message(clean_results))

        return "已达到最大步数限制。"


class PlanExecuteAgent(BaseAgent):
    """范式 2：Plan-and-Execute

    问题：SimpleAgent 在复杂任务上容易迷失 —— 它边走边看，没有全局视角。
    解决：在执行前先让 LLM 制定计划，然后按计划逐步执行。

    这个范式揭示了 Agent 设计的核心权衡：
    - 规划开销 vs 执行效率：规划消耗 token，但能减少执行中的试错
    - 计划刚性 vs 灵活性：严格按计划走可能错过更好的路径

    对应 Phase 1 的 planning_interval 机制，以及 LangGraph 的 Plan-and-Execute 模式。
    """

    PLANNER_SYSTEM = (
        "你是一个任务规划专家。将任务拆解为 3-5 个具体步骤。\n"
        "每个步骤应该是独立可执行的。简洁输出，每行一个步骤。\n"
        "不要编号，只输出步骤描述。"
    )

    EXECUTOR_SYSTEM = (
        "你是一个任务执行者。根据计划和已完成步骤的结果，执行当前步骤。\n"
        "如果需要工具（计算、搜索、数据分析），就调用工具。\n"
        "每一步的结论要简洁明确。"
    )

    SYNTHESIZER_SYSTEM = (
        "你是一个综合专家。将所有步骤的结果合并为一个连贯的最终答案。"
    )

    def run(self, task: str) -> str:
        self.steps = []

        # Phase 1: 规划
        console.print(f"\n  [bold cyan]═══ 阶段 1：规划 ═══[/bold cyan]")
        plan_response = self.client.send(
            messages=[{"role": "user", "content": f"任务: {task}"}],
            system=self.PLANNER_SYSTEM,
            max_tokens=512,
        )
        plan_text = plan_response.content[0].text if plan_response.content else ""
        plan_steps = [s.strip() for s in plan_text.strip().split("\n") if s.strip()]
        console.print(Panel(
            "\n".join(f"  {j+1}. {s}" for j, s in enumerate(plan_steps)),
            title="执行计划",
            border_style="cyan",
        ))

        # Phase 2: 逐步执行
        results: list[str] = []
        for idx, plan_step in enumerate(plan_steps):
            console.print(f"\n  [bold green]═══ 阶段 2：执行步骤 {idx+1}/{len(plan_steps)} ═══[/bold green]")

            context = ""
            if results:
                context = "前序步骤结果:\n" + "\n".join(
                    f"  步骤{j+1}: {r[:100]}" for j, r in enumerate(results)
                )

            step_result = SimpleAgent(self.client, max_steps=3).run(
                f"任务: {plan_step}\n\n{context}"
            )
            results.append(step_result)
            console.print(f"  [dim]步骤 {idx+1} 结果: {step_result[:150]}[/dim]")

        # Phase 3: 综合
        console.print(f"\n  [bold yellow]═══ 阶段 3：综合 ═══[/bold yellow]")
        synthesis_input = "\n".join(
            f"步骤 {j+1}: {r}" for j, r in enumerate(results)
        )
        synth_response = self.client.send(
            messages=[{"role": "user", "content": f"任务: {task}\n\n各步骤结果:\n{synthesis_input}"}],
            system=self.SYNTHESIZER_SYSTEM,
            max_tokens=1024,
        )
        final = synth_response.content[0].text if synth_response.content else ""
        return final


class ReflectionAgent(BaseAgent):
    """范式 3：Reflection（带自我反思的循环）

    问题：SimpleAgent 和 PlanExecuteAgent 都假设执行结果是对的。
         但如果某一步的结果质量不高，它们不会察觉。

    解决：在每次执行后加入反思步骤，让 Agent 评估自己的输出质量。
         不合格 → 重新执行，合格 → 继续。

    这对应 LangGraph 的 conditional edges + re-plan 循环。
    核心洞察：Agent 不仅需要"做事"的能力，还需要"判断自己做得好不好"的能力。

    Reflection 的三种模式：
    1. 自我反思（Self-Reflection）: Agent 检查自己的输出 —— 本文件的实现
    2. 外部反思（External Reflection）: 另一个 Agent 来审查 —— 02 文件演示
    3. 环境反思（Environment Reflection）: 通过工具执行结果来验证 —— 如幻觉检查
    """

    ACTOR_SYSTEM = (
        "你是一个任务执行者。基于用户任务执行。需要时使用工具。"
    )

    REFLECTOR_SYSTEM = (
        "你是一个质量评估者。评估以下执行结果：\n"
        "1. 是否完整回答了任务？\n"
        "2. 是否有事实错误？\n"
        "3. 是否清晰简洁？\n\n"
        "如果满意，回复 'PASS'。如果不满意，回复 'RETRY: <具体问题>'。"
    )

    def run(self, task: str, max_reflections: int = 3) -> str:
        self.steps = []

        for reflection_round in range(1, max_reflections + 1):
            console.print(f"\n  [bold cyan]═══ 反思轮次 {reflection_round}/{max_reflections} ═══[/bold cyan]")

            # 执行
            messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
            response = self.client.send(
                messages=messages,
                system=self.ACTOR_SYSTEM,
                tools=TOOLS_SCHEMA,
                max_tokens=1024,
            )
            text, tool_uses = self.client.extract_text_and_tools(response)
            output = text

            # 如果有工具调用，执行它们
            if tool_uses:
                tool_results = self.client.execute_tools(tool_uses)
                result_texts = [r["content"] for r in tool_results]
                output = text + "\n\n工具结果:\n" + "\n".join(result_texts)

            # 反思
            reflect_response = self.client.send(
                messages=[{"role": "user", "content": f"任务: {task}\n\n执行结果:\n{output}"}],
                system=self.REFLECTOR_SYSTEM,
                max_tokens=256,
            )
            verdict = reflect_response.content[0].text if reflect_response.content else "PASS"

            step = AgentStep(
                step_num=reflection_round,
                thinking=text[:200],
                final_answer=output[:200],
                token_usage={
                    "input": response.usage.input_tokens + reflect_response.usage.input_tokens,
                    "output": response.usage.output_tokens + reflect_response.usage.output_tokens,
                },
            )
            self.steps.append(step)

            console.print(f"  [yellow]反思判定: {verdict[:100]}[/yellow]")

            if verdict.strip().upper().startswith("PASS"):
                return output

            # 将反思作为反馈加入下一轮
            task = f"{task}\n\n上次尝试的问题（请修正）: {verdict}"

        return output


# ═══════════════════════════════════════════════════════════════
# 4. 运行演示
# ═══════════════════════════════════════════════════════════════

def demo_simple_agent(client: ClaudeClient):
    """演示 1：基础 ReAct 循环"""
    console.print(Panel(
        "[bold]范式 1: SimpleAgent（基础 ReAct 循环）[/bold]\n"
        "直接对应 Phase 1 ExecutionEngine 的 while 循环",
        border_style="yellow",
    ))

    agent = SimpleAgent(client, max_steps=8)

    tasks = [
        "帮我算一下 e^2 + sqrt(pi * 10) 约等于多少",
        "LangGraph 是什么？它的核心设计思想是什么？",
        "分析以下数据: [23, 45, 12, 67, 34, 89, 21, 56, 33, 78]",
    ]

    for task in tasks:
        console.print(f"\n[bold]任务:[/bold] {task}")
        result = agent.run(task)
        console.print(f"[bold]结果:[/bold] {result[:300]}")
        agent.print_trace()


def demo_plan_execute(client: ClaudeClient):
    """演示 2：Plan-and-Execute"""
    console.print(Panel(
        "[bold]范式 2: PlanExecuteAgent（先规划后执行）[/bold]\n"
        "对应 LangGraph 的 Plan-and-Execute 模式",
        border_style="yellow",
    ))

    agent = PlanExecuteAgent(client, max_steps=10)

    task = "分析 AI Agent 框架 LangGraph 和 CrewAI 的核心差异，给出选型建议"
    console.print(f"\n[bold]任务:[/bold] {task}")
    result = agent.run(task)
    console.print(Panel(result[:500], title="最终答案", border_style="green"))


def demo_reflection(client: ClaudeClient):
    """演示 3：Reflection Agent"""
    console.print(Panel(
        "[bold]范式 3: ReflectionAgent（带自我反思）[/bold]\n"
        "执行 → 自我评估 → PASS 或 RETRY",
        border_style="yellow",
    ))

    agent = ReflectionAgent(client, max_steps=8)

    task = "搜索 LangGraph 和 ReAct 模式的知识，分析它们之间的关系，给出一个清晰的定义"
    console.print(f"\n[bold]任务:[/bold] {task}")
    result = agent.run(task, max_reflections=2)
    console.print(Panel(result[:500], title="最终答案", border_style="green"))


def design_comparison():
    """设计对比总结"""
    console.print(Panel(
        "[bold]三种 Agent 循环范式的设计对比[/bold]\n\n"
        "┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐\n"
        "│ 维度            │ SimpleAgent      │ PlanExecuteAgent │ ReflectionAgent  │\n"
        "├─────────────────┼──────────────────┼──────────────────┼──────────────────┤\n"
        "│ 控制流          │ while 循环       │ 规划→执行→综合   │ 执行→评估→重试   │\n"
        "│ 终止条件        │ LLM 自主决定     │ 计划完成          │ 质量评估通过      │\n"
        "│ 适合场景        │ 简单查询/计算    │ 多步骤任务        │ 质量敏感任务      │\n"
        "│ Token 开销      │ 低               │ 中（规划+执行）   │ 高（多轮重试）    │\n"
        "│ 失败处理        │ 无（继续执行）   │ 步骤级重试        │ 整体重试          │\n"
        "│ Phase 1 对应    │ ExecutionEngine  │ planning_interval │ 无（Phase 3 新增）│\n"
        "│ LangGraph 对应  │ StateGraph(ReAct)│ Plan-and-Execute  │ Cond. Edges+Loop  │\n"
        "└─────────────────┴──────────────────┴──────────────────┴──────────────────┘\n\n"
        "[bold]Phase 1 vs Claude SDK vs LangGraph 的分层关系[/bold]\n\n"
        "Phase 1 (手写):\n"
        "  while step < max_steps:\n"
        "      thought = llm.think(...)\n"
        "      result = tool.execute(...)\n"
        "  → 全部手写，最大灵活性\n\n"
        "Claude SDK (API 工具层):\n"
        "  response = client.messages.create(...)\n"
        "  → 消息格式和工具 schema 标准化，循环逻辑仍需手写\n\n"
        "LangGraph (编排层):\n"
        "  graph.add_node(...) + graph.add_conditional_edges(...)\n"
        "  → 控制流也声明式管理，但学习成本最高",
        title="设计思考",
        border_style="green",
    ))


def run_demo():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]请设置 ANTHROPIC_API_KEY 环境变量[/red]")
        console.print("[dim]cp .env.example .env 并填入你的 API Key[/dim]")
        return

    console.print(Panel(
        "[bold]Claude SDK Agent 循环：三种范式[/bold]\n"
        "SimpleAgent（ReAct） / PlanExecuteAgent（规划） / ReflectionAgent（反思）\n"
        "从 Phase 1 的手写循环到 LangGraph 的声明式编排 —— 理解分层设计",
        title="01 Agent Loop — 三种范式",
        border_style="blue",
    ))

    client = ClaudeClient(api_key=api_key)

    demo_simple_agent(client)
    demo_plan_execute(client)
    demo_reflection(client)
    design_comparison()


if __name__ == "__main__":
    run_demo()
