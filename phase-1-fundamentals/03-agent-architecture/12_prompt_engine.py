"""
12_prompt_engine.py — 动态 Prompt 组装引擎

从 smolagents 和 langchain 源码中提炼 Prompt 管理的核心设计。
展示：模板系统、动态变量注入、消息历史管理、多角色 Prompt 组装。

对应文章第 4 章：Prompt 工程
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


# ── 1. 消息抽象 ─────────────────────────────────────────────────
# smolagents: ChatMessage(role, content) — role 包含 system/user/assistant/tool_response
# langchain:  SystemMessage / HumanMessage / AIMessage / ToolMessage

@dataclass
class Message:
    role: str  # system / user / assistant / tool
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# ── 2. Prompt 模板引擎 ──────────────────────────────────────────
# smolagents: Jinja2 模板 + populate_template()
# langchain:  PromptTemplate + ChatPromptTemplate
# 共性：模板 + 变量注入，将工具描述、指令、历史动态组装

SYSTEM_PROMPT_TEMPLATE = """你是一个智能助手，能够使用工具来完成任务。

## 可用工具

{tool_descriptions}

## 工作方式

每一步你需要：
1. **思考**：分析当前状况，决定下一步行动
2. **行动**：调用一个工具，或给出最终答案
3. **观察**：查看工具返回的结果

当你有足够信息时，使用 `final_answer` 给出最终答案。

{custom_instructions}"""

PLANNING_TEMPLATE = """基于当前任务和已有信息，制定下一步计划。

任务: {task}
已完成步骤: {completed_steps}
剩余步数: {remaining_steps}

请更新你的行动计划。"""


class PromptEngine:
    """
    Prompt 组装引擎。

    架构要点（从源码提炼）：
    1. System Prompt 是模板，工具描述和指令在运行时注入
    2. 消息历史按时间顺序累积，构成 LLM 的完整上下文
    3. 规划 Prompt 在特定步骤插入，引导 LLM 反思
    4. smolagents 用 Jinja2 模板，langchain 用 PromptTemplate
    """

    def __init__(self, system_template: str = SYSTEM_PROMPT_TEMPLATE):
        self.system_template = system_template
        self.messages: list[Message] = []

    def build_system_prompt(self, tool_descriptions: str, custom_instructions: str = "") -> str:
        """组装 System Prompt（运行时注入工具描述和自定义指令）"""
        return self.system_template.format(
            tool_descriptions=tool_descriptions,
            custom_instructions=custom_instructions,
        )

    def initialize(self, tool_descriptions: str, task: str, instructions: str = "") -> None:
        """初始化消息序列"""
        system_prompt = self.build_system_prompt(tool_descriptions, instructions)
        self.messages = [
            Message("system", system_prompt),
            Message("user", f"任务: {task}"),
        ]

    def add_assistant_message(self, thought: str, action: str, action_input: str) -> None:
        """添加 LLM 的思考和行动"""
        content = f"思考: {thought}\n行动: {action}({action_input})"
        self.messages.append(Message("assistant", content))

    def add_tool_response(self, observation: str) -> None:
        """添加工具执行结果"""
        self.messages.append(Message("tool", f"观察: {observation}"))

    def add_error(self, error: str) -> None:
        """添加错误信息（引导 LLM 重试）"""
        # smolagents 的错误消息模板：包含重试提示
        self.messages.append(Message("tool",
            f"错误: {error}\n请重试，注意不要重复之前的错误。如果多次失败，尝试完全不同的方法。"
        ))

    def add_planning_prompt(self, task: str, completed: int, remaining: int) -> None:
        """插入规划 Prompt"""
        planning = PLANNING_TEMPLATE.format(
            task=task, completed_steps=completed, remaining_steps=remaining,
        )
        self.messages.append(Message("user", planning))

    def get_messages(self) -> list[dict]:
        """导出为 LLM API 格式"""
        return [m.to_dict() for m in self.messages]

    def get_summary_messages(self) -> list[dict]:
        """导出精简版消息（跳过 system 和 planning，用于上下文压缩）"""
        return [
            m.to_dict() for m in self.messages
            if m.role not in ("system",) and "制定下一步计划" not in m.content
        ]

    def get_context_length(self) -> int:
        """估算上下文长度（字符数）"""
        return sum(len(m.content) for m in self.messages)


# ── 3. 演示 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = PromptEngine()

    # 模拟工具描述（两种风格）
    tool_desc_code = """def search(query: str) -> str:
    \"\"\"搜索互联网获取信息\"\"\"

def calculator(expression: str) -> str:
    \"\"\"计算数学表达式\"\"\""""

    # 初始化
    engine.initialize(
        tool_descriptions=tool_desc_code,
        task="北京今天的天气怎么样？如果温度超过 25 度，推荐一个室内活动。",
        instructions="回答时使用中文，语气友好。",
    )

    # 模拟多轮交互
    engine.add_assistant_message("需要查询天气", "search", "北京今天天气")
    engine.add_tool_response("北京今天晴，最高温度 28°C，最低 18°C")
    engine.add_assistant_message("温度 28°C > 25°C，需要推荐室内活动", "search", "北京室内活动推荐")
    engine.add_tool_response("推荐：国家博物馆、798艺术区、三里屯购物")
    engine.add_assistant_message(
        "信息足够了", "final_answer",
        "北京今天晴，28°C。温度较高，推荐室内活动：国家博物馆、798艺术区。"
    )

    # 输出完整消息序列
    print("=== 完整消息序列 ===\n")
    for i, msg in enumerate(engine.get_messages()):
        role_emoji = {"system": "⚙️", "user": "👤", "assistant": "🤖", "tool": "🔧"}
        print(f"[{i}] {role_emoji.get(msg['role'], '?')} {msg['role'].upper()}")
        # 截断长消息
        content = msg["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"    {content}\n")

    print(f"=== 上下文统计 ===")
    print(f"消息数: {len(engine.messages)}")
    print(f"估算字符数: {engine.get_context_length()}")
