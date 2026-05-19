"""
05_persistence_memory.py — LangGraph 状态持久化：Checkpointing 与跨会话记忆

设计思考：
Phase 1 的 Agent 是无状态的 —— 每次运行从零开始，对话历史只在内存中。
这在学习和 demo 中没问题，但生产环境需要：
1. 跨会话连续性 —— 用户关闭浏览器后回来，对话还在
2. 故障恢复 —— 服务重启后，进行中的任务能继续
3. 时间旅行调试 —— 回到任意历史状态，重放执行过程

LangGraph 的 Checkpointer 是解决这些问题的基础设施：
- 每个节点执行后自动保存状态快照
- 通过 thread_id 隔离不同会话
- 支持多种后端：InMemorySaver（开发）、SqliteSaver（单机）、PostgresSaver（生产）

这不只是"保存聊天记录"，而是保存整个图的执行状态 ——
包括中间节点的输出、条件边的路由决策、interrupt 的暂停点。

运行方式：
    python 05_persistence_memory.py
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()


# ── 1. 状态定义 ─────────────────────────────────────────────────

class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str       # 对话摘要（长对话时压缩历史）
    turn_count: int    # 对话轮次


# ── 2. LLM ──────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# ── 3. 节点函数 ─────────────────────────────────────────────────

def chat(state: ConversationState) -> dict:
    """对话节点：带摘要上下文的多轮对话"""
    system_parts = ["你是一个有记忆的 AI 助手。"]
    if state.get("summary"):
        system_parts.append(f"之前的对话摘要：{state['summary']}")

    response = llm.invoke([
        SystemMessage(content=" ".join(system_parts)),
        *state["messages"],
    ])
    return {
        "messages": [response],
        "turn_count": state.get("turn_count", 0) + 1,
    }


def maybe_summarize(state: ConversationState) -> dict:
    """摘要节点：对话超过阈值时压缩历史

    这是 LangGraph 处理长对话的模式：
    不是无限增长消息列表，而是定期压缩为摘要。
    Phase 4 会深入记忆系统设计，这里展示基本思路。
    """
    messages = state["messages"]
    if len(messages) <= 6:
        return {}

    # 保留最近 4 条消息，其余压缩为摘要
    old_messages = messages[:-4]
    old_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:100]}"
        for m in old_messages
    )

    response = llm.invoke([
        SystemMessage(content="将以下对话历史压缩为一段简短摘要，保留关键信息。"),
        HumanMessage(content=f"对话历史:\n{old_text}\n\n已有摘要: {state.get('summary', '无')}"),
    ])

    console.print(f"  [dim]对话压缩: {len(old_messages)} 条消息 → 摘要[/dim]")
    return {
        "summary": response.content,
        "messages": messages[-4:],  # 只保留最近的消息
    }


# ── 4. 路由 ─────────────────────────────────────────────────────

def should_summarize(state: ConversationState) -> str:
    if len(state["messages"]) > 6:
        return "summarize"
    return END


# ── 5. 组装图 ───────────────────────────────────────────────────

graph = StateGraph(ConversationState)
graph.add_node("chat", chat)
graph.add_node("summarize", maybe_summarize)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", should_summarize, {
    "summarize": "summarize",
    END: END,
})
graph.add_edge("summarize", END)

# Checkpointer：状态持久化的核心
# InMemorySaver 用于开发，生产环境换成 SqliteSaver 或 PostgresSaver
memory = InMemorySaver()
app = graph.compile(checkpointer=memory)


# ── 6. 运行演示 ─────────────────────────────────────────────────

def demo_multi_turn():
    """演示 1：多轮对话 + 状态持久化"""
    console.print(Panel(
        "[bold]演示 1：多轮对话持久化[/bold]\n"
        "同一个 thread_id 的对话共享状态",
        border_style="yellow",
    ))

    config = {"configurable": {"thread_id": "user-alice"}}

    conversations = [
        "你好，我叫 Alice，我是一个后端工程师",
        "我最近在学习 AI Agent 开发",
        "你还记得我的名字和职业吗？",
    ]

    for msg in conversations:
        console.print(f"\n[bold]用户:[/bold] {msg}")
        result = app.invoke(
            {"messages": [HumanMessage(content=msg)], "summary": "", "turn_count": 0},
            config,
        )
        console.print(f"[bold]AI:[/bold] {result['messages'][-1].content[:200]}")


def demo_thread_isolation():
    """演示 2：线程隔离 —— 不同 thread_id 的对话互不影响"""
    console.print(Panel(
        "[bold]演示 2：线程隔离[/bold]\n"
        "不同 thread_id 的对话状态完全独立",
        border_style="yellow",
    ))

    # Alice 的对话
    config_alice = {"configurable": {"thread_id": "user-alice-2"}}
    app.invoke(
        {"messages": [HumanMessage(content="我叫 Alice，喜欢 Python")], "summary": "", "turn_count": 0},
        config_alice,
    )

    # Bob 的对话
    config_bob = {"configurable": {"thread_id": "user-bob"}}
    app.invoke(
        {"messages": [HumanMessage(content="我叫 Bob，喜欢 Rust")], "summary": "", "turn_count": 0},
        config_bob,
    )

    # 验证隔离：Alice 的线程不知道 Bob
    result = app.invoke(
        {"messages": [HumanMessage(content="我喜欢什么编程语言？")], "summary": "", "turn_count": 0},
        config_alice,
    )
    console.print(f"\n[bold]Alice 线程问「我喜欢什么语言」:[/bold]")
    console.print(f"  {result['messages'][-1].content[:200]}")

    result = app.invoke(
        {"messages": [HumanMessage(content="我喜欢什么编程语言？")], "summary": "", "turn_count": 0},
        config_bob,
    )
    console.print(f"\n[bold]Bob 线程问「我喜欢什么语言」:[/bold]")
    console.print(f"  {result['messages'][-1].content[:200]}")


def demo_state_inspection():
    """演示 3：状态检查 —— 查看 Checkpointer 保存的完整状态"""
    console.print(Panel(
        "[bold]演示 3：状态检查（时间旅行）[/bold]\n"
        "Checkpointer 保存每一步的完整状态快照",
        border_style="yellow",
    ))

    config = {"configurable": {"thread_id": "inspect-demo"}}

    app.invoke(
        {"messages": [HumanMessage(content="第一轮对话")], "summary": "", "turn_count": 0},
        config,
    )
    app.invoke(
        {"messages": [HumanMessage(content="第二轮对话")], "summary": "", "turn_count": 0},
        config,
    )

    # 获取状态历史
    states = list(app.get_state_history(config))
    table = Table(title="状态历史（Checkpoint 快照）")
    table.add_column("序号", style="cyan")
    table.add_column("消息数", style="green")
    table.add_column("轮次", style="yellow")
    table.add_column("Checkpoint ID", style="dim")

    for i, state in enumerate(states[:5]):
        msg_count = len(state.values.get("messages", []))
        turn = state.values.get("turn_count", 0)
        checkpoint_id = state.config.get("configurable", {}).get("checkpoint_id", "")[:16]
        table.add_row(str(i), str(msg_count), str(turn), checkpoint_id + "...")

    console.print(table)


def run_demo():
    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[red]请设置 DEEPSEEK_API_KEY 环境变量[/red]")
        console.print("[dim]cp .env.example .env 并填入你的 API Key[/dim]")
        return
    console.print(Panel(
        "[bold]LangGraph 状态持久化[/bold]\n"
        "Checkpointer = 可暂停、可恢复、可回溯的 Agent\n"
        "从 Phase 1 的无状态 while 循环到有状态的图执行",
        title="05 Persistence & Memory",
        border_style="blue",
    ))

    demo_multi_turn()
    demo_thread_isolation()
    demo_state_inspection()

    console.print(Panel(
        "[bold]Checkpointing 的设计哲学[/bold]\n\n"
        "Phase 1 的 Agent:\n"
        "  memory = []  # 内存中的列表，进程结束就没了\n\n"
        "LangGraph 的 Agent:\n"
        "  checkpointer = SqliteSaver(db)  # 每步自动持久化\n"
        "  config = {'thread_id': 'user-123'}  # 按线程隔离\n\n"
        "Checkpointer 不只是「保存聊天记录」：\n"
        "  1. 保存完整图状态（所有节点的输出）\n"
        "  2. 支持 interrupt/resume（人机协作的基础）\n"
        "  3. 支持时间旅行（回到任意历史状态）\n"
        "  4. 支持分支（从某个状态分叉出不同路径）\n\n"
        "生产环境选择：\n"
        "  开发: InMemorySaver（快，重启丢失）\n"
        "  单机: SqliteSaver（持久化，无需额外服务）\n"
        "  生产: PostgresSaver（高可用，多实例共享）",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
