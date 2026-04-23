"""
10_execution_engine.py — 可扩展的 Agent 执行引擎

从 smolagents 和 langchain 源码中提炼出 Agent 执行循环的核心架构。
用 ~80 行代码展示：步骤抽象、终止判断、错误恢复、规划介入。

对应文章第 2 章：Agent 执行引擎
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── 1. 步骤抽象（Step Abstraction）──────────────────────────────
# smolagents: MemoryStep → ActionStep / PlanningStep / TaskStep
# langchain:  AgentAction / AgentFinish
# 共性：每一步都是一个不可变的数据记录

@dataclass
class Step:
    """所有步骤的基类"""
    step_number: int = 0

@dataclass
class ThoughtStep(Step):
    """LLM 的思考 + 行动决策"""
    thought: str = ""
    action: str = ""
    action_input: str = ""

@dataclass
class ObservationStep(Step):
    """工具执行后的观察结果"""
    observation: str = ""
    error: str | None = None

@dataclass
class FinalAnswer(Step):
    """终止步骤"""
    answer: Any = None


# ── 2. 执行引擎（Execution Engine）──────────────────────────────
# 核心循环：while not done → think → act → observe → repeat
# smolagents: MultiStepAgent._run_stream() 中的 while 循环
# langchain:  AgentExecutor._iter_next_step() 中的 while 循环

class ExecutionEngine:
    """
    Agent 执行引擎的最小实现。

    架构要点（从源码提炼）：
    1. 步骤计数 + 最大步数限制（防止无限循环）
    2. 每步产出不可变的 Step 记录（可追溯、可回放）
    3. 错误不终止循环，而是作为观察反馈给 LLM
    4. 规划步骤可按间隔插入（smolagents 的 planning_interval）
    """

    def __init__(self, think_fn, act_fn, max_steps: int = 10, planning_interval: int | None = None):
        self.think_fn = think_fn
        self.act_fn = act_fn
        self.max_steps = max_steps
        self.planning_interval = planning_interval
        self.memory: list[Step] = []

    def run(self, task: str) -> Any:
        """执行主循环"""
        print(f"\n{'='*60}")
        print(f"任务: {task}")
        print(f"{'='*60}")

        for step_num in range(1, self.max_steps + 1):
            if self.planning_interval and step_num % self.planning_interval == 1:
                print(f"\n📋 [规划] 第 {step_num} 步，暂停反思...")
                self._reflect(task, step_num)

            thought_step = self.think_fn(task, self.memory, step_num)
            self.memory.append(thought_step)
            print(f"\n🧠 Step {step_num} | 思考: {thought_step.thought}")
            print(f"   行动: {thought_step.action}({thought_step.action_input})")

            if thought_step.action == "final_answer":
                answer = FinalAnswer(step_number=step_num, answer=thought_step.action_input)
                self.memory.append(answer)
                print(f"\n✅ 最终答案: {answer.answer}")
                return answer.answer

            try:
                result = self.act_fn(thought_step.action, thought_step.action_input)
                obs = ObservationStep(step_number=step_num, observation=str(result))
            except Exception as e:
                obs = ObservationStep(step_number=step_num, error=str(e))
                print(f"   ⚠️ 错误: {obs.error}")

            self.memory.append(obs)
            print(f"   观察: {obs.observation or obs.error}")

        print(f"\n⏰ 达到最大步数 {self.max_steps}，强制生成答案")
        return self._force_final_answer(task)

    def _reflect(self, task: str, step_num: int):
        completed = [s for s in self.memory if isinstance(s, ObservationStep) and not s.error]
        print(f"   已完成 {len(completed)} 个有效步骤")

    def _force_final_answer(self, task: str) -> str:
        observations = [s.observation for s in self.memory if isinstance(s, ObservationStep) and s.observation]
        return f"基于已有信息的总结: {'; '.join(observations[-3:])}"


# ── 3. 演示 ─────────────────────────────────────────────────────

def demo_think(task: str, memory: list[Step], step_num: int) -> ThoughtStep:
    """模拟 LLM 思考（实际中由 LLM 生成）"""
    if step_num == 1:
        return ThoughtStep(step_num, "需要先搜索相关信息", "search", "AI Agent 架构设计")
    elif step_num == 2:
        return ThoughtStep(step_num, "需要计算一个数据", "calculator", "1000 * 0.8")
    else:
        prev_obs = [s for s in memory if isinstance(s, ObservationStep)]
        summary = "; ".join(o.observation for o in prev_obs if o.observation)
        return ThoughtStep(step_num, "信息足够了，给出答案", "final_answer", summary)


def demo_act(action: str, action_input: str) -> str:
    """模拟工具执行"""
    mock_results = {
        "search": "搜索结果: Agent 架构包含执行引擎、工具系统、记忆管理三大模块",
        "calculator": "计算结果: 800.0",
    }
    if action not in mock_results:
        raise ValueError(f"未知工具: {action}")
    return mock_results[action]


if __name__ == "__main__":
    engine = ExecutionEngine(
        think_fn=demo_think,
        act_fn=demo_act,
        max_steps=10,
        planning_interval=3,
    )
    result = engine.run("分析 AI Agent 的核心架构组件")
    print(f"\n{'='*60}")
    print(f"执行完成，共 {len(engine.memory)} 个步骤记录")
