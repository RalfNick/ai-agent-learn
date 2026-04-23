"""
14_memory_context.py — 记忆与上下文管理

从 smolagents 的 AgentMemory / MemoryStep / CallbackRegistry 中提炼记忆管理的核心设计。
展示：步骤存储、summary_mode 上下文压缩、token 估算与预算控制、状态传递、步骤回调。

对应文章第 6 章：状态与记忆
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


# ── 1. 步骤类型（复用 10_execution_engine 的设计）─────────────────
# smolagents: MemoryStep → ActionStep / PlanningStep / TaskStep / SystemPromptStep
# 这里定义完整的步骤层级，支持 summary_mode

@dataclass
class MemoryStep:
    """所有步骤的基类"""
    step_number: int = 0

    def to_messages(self, summary_mode: bool = False) -> list[dict]:
        return []

@dataclass
class SystemPromptStep(MemoryStep):
    """系统提示词快照"""
    content: str = ""

    def to_messages(self, summary_mode: bool = False) -> list[dict]:
        if summary_mode:
            return []
        return [{"role": "system", "content": self.content}]

@dataclass
class TaskStep(MemoryStep):
    """用户任务"""
    task: str = ""

    def to_messages(self, summary_mode: bool = False) -> list[dict]:
        return [{"role": "user", "content": f"任务: {self.task}"}]

@dataclass
class ActionStep(MemoryStep):
    """LLM 思考 + 工具调用 + 观察结果（一次完整的 Think-Act-Observe）"""
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""
    error: str | None = None

    def to_messages(self, summary_mode: bool = False) -> list[dict]:
        msgs = []
        if not summary_mode and self.thought:
            msgs.append({"role": "assistant", "content": f"思考: {self.thought}\n行动: {self.action}({self.action_input})"})
        if self.observation:
            msgs.append({"role": "tool", "content": f"观察: {self.observation}"})
        if self.error:
            msgs.append({"role": "tool", "content": f"错误: {self.error}"})
        return msgs

@dataclass
class PlanningStep(MemoryStep):
    """规划/反思步骤"""
    plan: str = ""

    def to_messages(self, summary_mode: bool = False) -> list[dict]:
        if summary_mode:
            return []
        return [{"role": "assistant", "content": f"[规划] {self.plan}"}]

@dataclass
class FinalAnswerStep(MemoryStep):
    """最终答案"""
    answer: Any = None

    def to_messages(self, summary_mode: bool = False) -> list[dict]:
        return [{"role": "assistant", "content": f"最终答案: {self.answer}"}]


# ── 2. 步骤回调注册表 ─────────────────────────────────────────────
# smolagents: CallbackRegistry — 按步骤类型注册回调，步骤完成时触发

class CallbackRegistry:
    """步骤回调注册表 — 按类型分发"""

    def __init__(self):
        self._callbacks: dict[type, list[Callable]] = {}

    def register(self, step_cls: type, callback: Callable) -> None:
        self._callbacks.setdefault(step_cls, []).append(callback)

    def fire(self, step: MemoryStep) -> None:
        for cls in type(step).__mro__:
            for cb in self._callbacks.get(cls, []):
                cb(step)


# ── 3. Agent 记忆管理 ─────────────────────────────────────────────
# smolagents: AgentMemory — system_prompt + steps + state
# 核心能力：存储、压缩、估算、序列化、回放

class AgentMemory:
    """
    Agent 记忆管理器。

    设计要点（从 smolagents AgentMemory 提炼）：
    - system_prompt 单独存储，可随时更新
    - steps 是追加式列表，不可变记录
    - state 是步骤间的共享状态字典（smolagents self.state）
    - summary_mode 压缩上下文（去掉 system prompt 和 planning）
    - callbacks 在每步追加时触发
    """

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = SystemPromptStep(content=system_prompt)
        self.steps: list[MemoryStep] = []
        self.state: dict[str, Any] = {}
        self.callbacks = CallbackRegistry()

    def add_step(self, step: MemoryStep) -> None:
        self.steps.append(step)
        self.callbacks.fire(step)

    def to_messages(self, summary_mode: bool = False) -> list[dict]:
        msgs = self.system_prompt.to_messages(summary_mode=summary_mode)
        for step in self.steps:
            msgs.extend(step.to_messages(summary_mode=summary_mode))
        return msgs

    def estimate_tokens(self) -> int:
        total_chars = sum(len(m["content"]) for m in self.to_messages())
        return total_chars // 2

    def trim_to_budget(self, max_tokens: int, keep_last: int = 3) -> int:
        """截断旧步骤的观察内容以控制 token 预算，返回截断数量"""
        trimmed = 0
        while self.estimate_tokens() > max_tokens:
            trimmable = [
                s for s in self.steps[:-keep_last]
                if isinstance(s, ActionStep) and s.observation and len(s.observation) > 100
            ]
            if not trimmable:
                break
            oldest = trimmable[0]
            oldest.observation = oldest.observation[:50] + "...[已截断]"
            trimmed += 1
        return trimmed

    def get_succinct_steps(self) -> list[dict]:
        results = []
        for s in self.steps:
            d = {"type": type(s).__name__, "step": s.step_number}
            if isinstance(s, ActionStep):
                d["action"] = s.action
                d["has_error"] = s.error is not None
            elif isinstance(s, TaskStep):
                d["task"] = s.task[:50]
            results.append(d)
        return results

    def replay(self) -> None:
        icons = {"TaskStep": "📋", "ActionStep": "🔄", "PlanningStep": "📐", "FinalAnswerStep": "✅"}
        for step in self.steps:
            name = type(step).__name__
            icon = icons.get(name, "?")
            print(f"  [{step.step_number}] {icon} {name}", end="")
            if isinstance(step, ActionStep):
                print(f" | {step.action}({step.action_input}) → {(step.observation or step.error or '')[:60]}")
            elif isinstance(step, TaskStep):
                print(f" | {step.task[:60]}")
            elif isinstance(step, PlanningStep):
                print(f" | {step.plan[:60]}")
            elif isinstance(step, FinalAnswerStep):
                print(f" | {step.answer}")
            else:
                print()

    def reset(self) -> None:
        self.steps.clear()
        self.state.clear()

    def to_json(self) -> str:
        data = {
            "system_prompt": self.system_prompt.content,
            "state": {k: str(v) for k, v in self.state.items()},
            "steps": [
                {"type": type(s).__name__, **{k: v for k, v in asdict(s).items() if v is not None}}
                for s in self.steps
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


# ── 4. 演示 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    memory = AgentMemory(system_prompt="你是一个智能助手，能够使用工具完成任务。")

    # 注册回调：监控每一步
    memory.callbacks.register(ActionStep, lambda s: print(f"  [回调] ActionStep #{s.step_number}: {s.action}"))
    memory.callbacks.register(FinalAnswerStep, lambda s: print(f"  [回调] 任务完成: {s.answer}"))

    # 添加任务
    memory.add_step(TaskStep(step_number=0, task="查询北京天气，如果超过25度推荐室内活动"))

    # 模拟多步执行 + state 传值
    memory.add_step(ActionStep(
        step_number=1, thought="需要查天气", action="search", action_input="北京天气",
        observation="北京今天晴，28°C",
    ))
    memory.state["weather"] = "北京今天晴，28°C"

    memory.add_step(PlanningStep(step_number=2, plan="温度28°C > 25°C，需要搜索室内活动"))

    memory.add_step(ActionStep(
        step_number=3, thought="搜索室内活动", action="search", action_input="北京室内活动",
        observation="推荐：国家博物馆、798艺术区、三里屯购物中心",
    ))
    memory.state["activities"] = "国家博物馆、798艺术区"

    memory.add_step(FinalAnswerStep(
        step_number=4, answer=f"天气: {memory.state['weather']}。推荐: {memory.state['activities']}",
    ))

    # 展示 state
    print(f"\n=== State 状态字典 ===")
    for k, v in memory.state.items():
        print(f"  {k}: {v}")

    # 对比 normal vs summary_mode
    print(f"\n=== Normal 模式消息（{len(memory.to_messages())} 条）===")
    for m in memory.to_messages():
        print(f"  [{m['role']}] {m['content'][:80]}")

    summary_msgs = memory.to_messages(summary_mode=True)
    print(f"\n=== Summary 模式消息（{len(summary_msgs)} 条，去掉 system + planning + thought）===")
    for m in summary_msgs:
        print(f"  [{m['role']}] {m['content'][:80]}")

    # Token 估算
    print(f"\n=== 上下文管理 ===")
    print(f"估算 token 数: {memory.estimate_tokens()}")
    trimmed = memory.trim_to_budget(max_tokens=100, keep_last=2)
    print(f"截断了 {trimmed} 个旧观察，当前 token: {memory.estimate_tokens()}")

    # 回放
    print(f"\n=== 执行回放 ===")
    memory.replay()

    # 精简步骤
    print(f"\n=== 精简步骤 ===")
    for s in memory.get_succinct_steps():
        print(f"  {s}")

    # JSON 序列化
    print(f"\n=== JSON 序列化（前 300 字符）===")
    print(memory.to_json()[:300] + "...")
