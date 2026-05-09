"""
02_human_in_the_loop.py — LangGraph 人机协作：interrupt 与审批工作流

设计思考：
Phase 1 的 ExecutionEngine 是全自动的 —— 一旦启动，跑到结束或超时。
这在 demo 里没问题，但生产环境中 Agent 经常需要"暂停等人确认"：
- 执行高风险操作前（删除数据、发送邮件、调用付费 API）
- 结果不确定时（让人类选择方向）
- 合规要求（审批流程）

LangGraph 的 interrupt() 是一等公民设计：
- 不是事后补丁，而是图执行模型的核心能力
- 依赖 Checkpointer：中断时保存完整状态，恢复时从断点继续
- 通过 Command(resume=value) 传递人类输入

这揭示了一个关键架构决策：
  无状态 Agent（Phase 1）无法实现真正的人机协作
  有状态 Agent（LangGraph + Checkpointer）才能"暂停-等待-恢复"

运行方式：
    python 02_human_in_the_loop.py
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
from langgraph.types import interrupt, Command
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

load_dotenv()
console = Console()


# ── 1. 状态定义 ─────────────────────────────────────────────────

class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str           # LLM 生成的草稿
    human_feedback: str   # 人类反馈
    approved: bool        # 是否通过审批
    final_output: str     # 最终输出


# ── 2. 节点函数 ─────────────────────────────────────────────────

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


def generate_draft(state: ReviewState) -> dict:
    """生成草稿节点"""
    response = llm.invoke([
        SystemMessage(content="你是一个专业的文案撰写助手。根据用户需求生成简洁的文案草稿。"),
        *state["messages"],
    ])
    return {"draft": response.content}


def human_review(state: ReviewState) -> dict:
    """人类审核节点 —— 这里是 interrupt 的核心用法

    interrupt() 做了三件事：
    1. 暂停图执行
    2. 把参数（审核内容）返回给调用方
    3. 等待 Command(resume=value) 恢复，value 成为 interrupt() 的返回值
    """
    console.print(Panel(state["draft"], title="待审核草稿", border_style="yellow"))

    # interrupt 暂停执行，等待人类输入
    # 在生产环境中，这个暂停可以持续数小时甚至数天
    feedback = interrupt({
        "type": "review_request",
        "draft": state["draft"],
        "prompt": "请审核这份草稿。输入修改意见，或输入 'approve' 批准。",
    })

    approved = feedback.strip().lower() in ("approve", "approved", "ok", "通过", "批准")
    return {
        "human_feedback": feedback,
        "approved": approved,
    }


def revise_draft(state: ReviewState) -> dict:
    """根据人类反馈修改草稿"""
    response = llm.invoke([
        SystemMessage(content="根据反馈修改文案草稿。保持简洁。"),
        HumanMessage(content=f"原始草稿:\n{state['draft']}\n\n修改意见:\n{state['human_feedback']}"),
    ])
    return {"draft": response.content}


def finalize(state: ReviewState) -> dict:
    """定稿节点"""
    return {"final_output": state["draft"]}


# ── 3. 路由函数 ─────────────────────────────────────────────────

def check_approval(state: ReviewState) -> str:
    """审核通过 → 定稿；未通过 → 修改"""
    if state.get("approved"):
        return "finalize"
    return "revise"


# ── 4. 组装图 ───────────────────────────────────────────────────
# 这个图展示了 LangGraph 处理人机协作的模式：
#   生成 → 人类审核 → (通过 → 定稿 | 不通过 → 修改 → 再审核)
# 注意 revise → human_review 形成循环，可以多轮修改

graph = StateGraph(ReviewState)

graph.add_node("generate", generate_draft)
graph.add_node("review", human_review)
graph.add_node("revise", revise_draft)
graph.add_node("finalize", finalize)

graph.add_edge(START, "generate")
graph.add_edge("generate", "review")
graph.add_conditional_edges("review", check_approval, {
    "finalize": "finalize",
    "revise": "revise",
})
graph.add_edge("revise", "review")  # 修改后再次审核（循环）
graph.add_edge("finalize", END)

# Checkpointer 是 interrupt 的前提 —— 没有状态持久化就无法暂停/恢复
memory = InMemorySaver()
app = graph.compile(checkpointer=memory)


# ── 5. 运行演示 ─────────────────────────────────────────────────

def run_demo():
    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[red]请设置 DEEPSEEK_API_KEY 环境变量[/red]")
        console.print("[dim]cp .env.example .env 并填入你的 API Key[/dim]")
        return
    console.print(Panel(
        "[bold]LangGraph 人机协作[/bold]\n"
        "interrupt() + Checkpointer = 可暂停、可恢复的 Agent\n"
        "演示：文案生成 → 人类审核 → 修改/批准 循环",
        title="02 Human-in-the-Loop",
        border_style="blue",
    ))

    config = {"configurable": {"thread_id": "review-session-1"}}

    # 第一次调用：生成草稿 → 到 human_review 节点时 interrupt
    console.print("\n[bold yellow]步骤 1：生成草稿并等待审核[/bold yellow]")
    task = "写一段关于 AI Agent 技术趋势的社交媒体文案，100字以内"

    result = app.invoke(
        {"messages": [HumanMessage(content=task)], "draft": "", "human_feedback": "",
         "approved": False, "final_output": ""},
        config,
    )

    # 图在 interrupt 处暂停，进入交互循环
    max_rounds = 3
    for round_num in range(1, max_rounds + 1):
        console.print(f"\n[bold cyan]审核轮次 {round_num}/{max_rounds}[/bold cyan]")
        feedback = Prompt.ask("你的反馈（输入 'approve' 批准，或输入修改意见）")

        # Command(resume=value) 恢复执行，value 成为 interrupt() 的返回值
        result = app.invoke(Command(resume=feedback), config)

        # 检查是否已定稿（图执行完毕）
        if result.get("final_output"):
            console.print(Panel(
                result["final_output"],
                title="最终定稿",
                border_style="green",
            ))
            break
    else:
        console.print("[red]达到最大审核轮次，流程结束[/red]")

    # 设计思考总结
    console.print(Panel(
        "[bold]interrupt 的设计哲学[/bold]\n\n"
        "Phase 1 的 Agent 是「全自动」的：\n"
        "  启动 → 跑完 → 返回结果（中间无法介入）\n\n"
        "LangGraph 的 Agent 是「可暂停」的：\n"
        "  启动 → 运行 → interrupt() 暂停 → 等待人类 → resume 恢复\n\n"
        "这需要两个基础设施：\n"
        "  1. Checkpointer — 暂停时保存完整状态\n"
        "  2. Thread ID — 恢复时找到正确的状态\n\n"
        "生产场景：\n"
        "  - 高风险操作审批（删除、支付、发送）\n"
        "  - 多人协作流程（A 生成 → B 审核 → C 执行）\n"
        "  - 长时间异步任务（提交后几小时再审批）",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
