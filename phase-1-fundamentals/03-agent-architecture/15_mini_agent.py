"""
15_mini_agent.py — 集成式迷你 Agent 框架

将前面的模块（工具系统、Prompt 引擎、记忆管理）组装成一个完整的 Agent。
展示：模块集成、完整执行循环、Agent-as-Tool 多 Agent 模式。

对应文章第 7 章：架构全景图
"""

from __future__ import annotations
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable
from abc import ABC, abstractmethod

# ── 动态导入（文件名以数字开头，用 importlib）─────────────────────
# 导入失败时 inline 重定义最小版本，保证可独立运行

def _import(module_name: str):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    return importlib.import_module(module_name)

try:
    _tools = _import("11_tool_system")
    Tool, ToolRegistry, tool = _tools.Tool, _tools.ToolRegistry, _tools.tool
except Exception:
    raise ImportError("请确保 11_tool_system.py 在同一目录下")

try:
    _prompt = _import("12_prompt_engine")
    PromptEngine, Message = _prompt.PromptEngine, _prompt.Message
except Exception:
    raise ImportError("请确保 12_prompt_engine.py 在同一目录下")

try:
    _memory = _import("14_memory_context")
    AgentMemory = _memory.AgentMemory
    ActionStep, TaskStep, PlanningStep = _memory.ActionStep, _memory.TaskStep, _memory.PlanningStep
    FinalAnswerStep = _memory.FinalAnswerStep
except Exception:
    raise ImportError("请确保 14_memory_context.py 在同一目录下")


# ── 1. MiniAgent — 集成式 Agent ──────────────────────────────────
# smolagents: MultiStepAgent 组合了 model + tools + memory + prompt
# 这里把 11（工具）、12（Prompt）、14（记忆）组装成完整 Agent

class MiniAgent:
    """
    迷你 Agent 框架 — 组装所有模块。

    设计要点（从 smolagents MultiStepAgent 提炼）：
    - ToolRegistry 管理工具，生成描述注入 Prompt
    - PromptEngine 组装消息序列
    - AgentMemory 存储步骤 + state + 回调
    - think_fn 注入 LLM 决策逻辑（可替换为真实 LLM）
    - __call__ 实现 Agent-as-Tool 接口
    """

    def __init__(
        self,
        name: str,
        description: str,
        tools: list[Tool],
        think_fn: Callable,
        max_steps: int = 10,
        planning_interval: int | None = None,
        instructions: str = "",
    ):
        self.name = name
        self.description = description
        self.think_fn = think_fn
        self.max_steps = max_steps
        self.planning_interval = planning_interval

        self.tool_registry = ToolRegistry()
        self.tool_registry.register_batch(tools)
        self.prompt_engine = PromptEngine()
        self.memory = AgentMemory(system_prompt="")

        self._instructions = instructions

    def run(self, task: str, additional_args: dict | None = None) -> Any:
        self.memory.reset()
        if additional_args:
            self.memory.state.update(additional_args)
            task += f"\n附加参数: {additional_args}"

        tool_desc = self.tool_registry.to_prompt(style="code")
        system_prompt = self.prompt_engine.build_system_prompt(tool_desc, self._instructions)
        self.memory.system_prompt.content = system_prompt
        self.memory.add_step(TaskStep(step_number=0, task=task))

        print(f"\n{'='*50}")
        print(f"[{self.name}] 任务: {task[:60]}")
        print(f"{'='*50}")

        for step_num in range(1, self.max_steps + 1):
            if self.planning_interval and step_num > 1 and (step_num - 1) % self.planning_interval == 0:
                plan = f"已完成 {step_num-1} 步，评估进展..."
                self.memory.add_step(PlanningStep(step_number=step_num, plan=plan))
                print(f"  📋 [规划] {plan}")

            messages = self.memory.to_messages()
            thought_step = self.think_fn(task, messages, step_num, self.tool_registry.tool_names)
            self.memory.add_step(thought_step)
            print(f"  🧠 Step {step_num} | {thought_step.action}({thought_step.action_input})")

            if thought_step.action == "final_answer":
                answer = thought_step.action_input
                self.memory.add_step(FinalAnswerStep(step_number=step_num, answer=answer))
                print(f"  ✅ 最终答案: {answer[:80]}")
                return answer

            try:
                if self.tool_registry.has(thought_step.action):
                    result = self.tool_registry.execute(thought_step.action, **_parse_args(thought_step.action_input))
                else:
                    result = f"未知工具: {thought_step.action}"
                thought_step.observation = str(result)
                self.memory.state[f"step_{step_num}"] = str(result)
            except Exception as e:
                thought_step.error = str(e)
                print(f"  ⚠️ 错误: {e}")

            print(f"     → {(thought_step.observation or thought_step.error or '')[:80]}")

        return f"[超时] 基于已有信息的总结"

    def __call__(self, task: str, additional_args: dict | None = None, **kwargs) -> str:
        result = self.run(task, additional_args)
        report = f"[{self.name} 报告] {result}"
        return report


def _parse_args(action_input: str) -> dict:
    if "=" in action_input:
        pairs = action_input.split(",")
        return {p.split("=")[0].strip(): p.split("=")[1].strip().strip("'\"") for p in pairs if "=" in p}
    return {"task": action_input} if action_input else {}


# ── 2. AgentTool — Agent 作为 Tool ───────────────────────────────
# smolagents: ManagedAgent 把子 Agent 包装成 Tool 接口
# Supervisor 通过 ToolRegistry 调用子 Agent，和调用普通工具一样

class AgentTool(Tool):
    """把 MiniAgent 包装为 Tool，实现 Agent-as-Tool 模式"""

    def __init__(self, agent: MiniAgent):
        self.name = agent.name
        self.description = agent.description
        self.parameters = {
            "task": {"type": "string", "description": "分配给该 Agent 的子任务"},
        }
        self.return_type = "string"
        self._agent = agent

    def forward(self, task: str, **kwargs) -> str:
        return self._agent(task, additional_args=kwargs.get("additional_args"))


# ── 3. 演示工具 ──────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """搜索互联网获取信息
    query: 搜索关键词
    """
    mock = {
        "AI Agent 市场": "AI Agent 市场预计 2026 年达到 500 亿美元",
        "AI Agent 增长率": "年增长率约 35%，企业自动化是主要驱动力",
    }
    for k, v in mock.items():
        if k in query:
            return v
    return f"搜索结果: 关于 '{query}' 的信息"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式
    expression: 数学表达式
    """
    return f"计算结果: {expression} = 42"


# ── 4. 演示 ─────────────────────────────────────────────────────

def make_think_fn(script: list[tuple[str, str]]):
    """创建一个按脚本执行的 think_fn（模拟 LLM 决策）"""
    idx = [0]
    def think_fn(task, messages, step_num, tool_names):
        if idx[0] < len(script):
            action, action_input = script[idx[0]]
            idx[0] += 1
        else:
            action, action_input = "final_answer", "基于已有信息的总结"
        return ActionStep(
            step_number=step_num, thought=f"第{step_num}步决策",
            action=action, action_input=action_input,
        )
    return think_fn


if __name__ == "__main__":
    # ── Demo 1: 单 Agent 全集成 ──────────────────────────────────
    print("=" * 60)
    print("Demo 1: 单 Agent — 工具 + 记忆 + Prompt 全集成")
    print("=" * 60)

    agent = MiniAgent(
        name="assistant",
        description="通用助手",
        tools=[web_search, calculator],
        think_fn=make_think_fn([
            ("web_search", "query=AI Agent 市场"),
            ("calculator", "expression=500 * 0.35"),
            ("final_answer", "AI Agent 市场 500 亿美元，年增长 35%"),
        ]),
        max_steps=10,
        planning_interval=3,
    )

    result = agent.run("分析 AI Agent 市场规模")

    print(f"\n--- 记忆回放 ---")
    agent.memory.replay()
    print(f"State: {agent.memory.state}")
    print(f"Token 估算: {agent.memory.estimate_tokens()}")

    # ── Demo 2: 多 Agent（Agent-as-Tool）─────────────────────────
    print("\n" + "=" * 60)
    print("Demo 2: 多 Agent — Supervisor 调用 Agent-as-Tool")
    print("=" * 60)

    researcher = MiniAgent(
        name="researcher", description="搜索和收集信息的研究员",
        tools=[web_search],
        think_fn=make_think_fn([
            ("web_search", "query=AI Agent 增长率"),
            ("final_answer", "年增长率约 35%"),
        ]),
    )

    analyst = MiniAgent(
        name="analyst", description="分析数据并提供洞察的分析师",
        tools=[calculator],
        think_fn=make_think_fn([
            ("calculator", "expression=500 * 1.35"),
            ("final_answer", "预计明年市场规模 675 亿美元"),
        ]),
    )

    researcher_tool = AgentTool(researcher)
    analyst_tool = AgentTool(analyst)

    supervisor = MiniAgent(
        name="supervisor", description="任务调度主管",
        tools=[researcher_tool, analyst_tool],
        think_fn=make_think_fn([
            ("researcher", "task=调研 AI Agent 行业增长率"),
            ("analyst", "task=预测明年市场规模"),
            ("final_answer", "研究完成：AI Agent 年增长 35%，明年预计 675 亿美元"),
        ]),
    )

    result = supervisor.run("全面分析 AI Agent 行业趋势")

    print(f"\n--- Supervisor 记忆回放 ---")
    supervisor.memory.replay()
    print(f"\n已注册工具: {supervisor.tool_registry.tool_names}")
    print(f"（其中 researcher 和 analyst 是 Agent-as-Tool）")
