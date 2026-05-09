"""
01_state_graph_basics.py — LangGraph 状态图基础：从 while 循环到有向图

设计思考：
Phase 1 中我们手写了 ExecutionEngine，核心是一个 while 循环：
    while not done: think → act → observe → repeat

LangGraph 的核心洞察是：Agent 工作流不只是循环，而是有向图。
- 节点（Node）= 处理函数（对应 Phase 1 的 think_fn / act_fn）
- 边（Edge）= 控制流（对应 Phase 1 while 循环里的 if/else）
- 状态（State）= 节点间的数据契约（对应 Phase 1 的 memory: list[Step]）

这个转变带来三个关键优势：
1. 可视化：图结构天然可画出来，while 循环不行
2. 可调试：每个节点独立，可以单独测试和替换
3. 可扩展：加分支、加并行，改边就行，不用改循环逻辑

运行方式：
    cp .env.example .env  # 填入 API Key
    pip install -r requirements.txt
    python 01_state_graph_basics.py
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()

# ── 1. 定义状态（State）─────────────────────────────────────────
# Phase 1 中，状态是 memory: list[Step]，隐式地在 while 循环中传递
# LangGraph 要求显式定义状态 schema —— 这是"状态优先"设计的核心
# 每个节点读写同一个 State，通过 Annotated + reducer 控制合并策略

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages 是 reducer：追加而非覆盖


# ── 2. 定义工具 ─────────────────────────────────────────────────
# LangGraph 的工具定义复用 LangChain 的 @tool 装饰器
# 与 Phase 1 手写的 ToolRegistry 不同，这里工具自带 schema 生成

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。支持加减乘除和幂运算。"""
    import ast
    import operator

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
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
        raise ValueError(f"不支持的运算: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_word_length(word: str) -> str:
    """获取一个单词或句子的字符长度。"""
    return f"'{word}' 的长度是 {len(word)} 个字符"


tools = [calculate, get_word_length]
tool_map = {t.name: t for t in tools}


# ── 3. 创建 LLM（绑定工具）──────────────────────────────────────

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
).bind_tools(tools)


# ── 4. 定义节点函数 ─────────────────────────────────────────────
# Phase 1 的 ExecutionEngine 把 think 和 act 作为回调传入
# LangGraph 中，每个节点就是一个普通函数：接收 State，返回 State 更新

def call_model(state: AgentState) -> dict:
    """模型节点：调用 LLM，可能产生工具调用或最终回答"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def call_tools(state: AgentState) -> dict:
    """工具节点：执行 LLM 请求的工具调用，返回结果"""
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        tool_fn = tool_map[call["name"]]
        result = tool_fn.invoke(call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )
    return {"messages": results}


# ── 5. 定义路由（条件边）────────────────────────────────────────
# Phase 1 中，"是否继续循环"的判断藏在 while 条件里
# LangGraph 把路由逻辑提取为独立函数，挂在条件边上 —— 显式、可测试

def should_continue(state: AgentState) -> str:
    """路由函数：LLM 输出有工具调用 → 走 tools 节点；否则 → 结束"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ── 6. 组装图 ───────────────────────────────────────────────────
# 这是 LangGraph 的核心：用声明式的方式描述 Agent 的控制流
# 对比 Phase 1 的 ExecutionEngine.__init__()，那里是命令式的回调注册

graph = StateGraph(AgentState)

# 添加节点
graph.add_node("model", call_model)
graph.add_node("tools", call_tools)

# 添加边：START → model（入口）
graph.add_edge(START, "model")

# 条件边：model 之后，根据是否有工具调用决定走向
graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})

# 工具执行完 → 回到 model（形成循环，这就是 ReAct 的 loop）
graph.add_edge("tools", "model")

# 编译图（类似 TensorFlow 的 compile，做结构校验）
agent = graph.compile()


# ── 7. 运行演示 ─────────────────────────────────────────────────

def run_demo():
    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[red]请设置 DEEPSEEK_API_KEY 环境变量[/red]")
        console.print("[dim]cp .env.example .env 并填入你的 API Key[/dim]")
        return
    console.print(Panel(
        "[bold]LangGraph 状态图基础[/bold]\n"
        "从 Phase 1 的 while 循环到 LangGraph 的有向图\n"
        "同样的 ReAct 模式，不同的表达方式",
        title="01 State Graph Basics",
        border_style="blue",
    ))

    # 展示图结构
    table = Table(title="图结构：节点与边")
    table.add_column("组件", style="cyan")
    table.add_column("类型", style="green")
    table.add_column("说明")
    table.add_row("model", "节点", "调用 LLM，产生回答或工具调用")
    table.add_row("tools", "节点", "执行工具，返回结果")
    table.add_row("START → model", "边", "入口")
    table.add_row("model → tools/END", "条件边", "有工具调用 → tools；否则 → END")
    table.add_row("tools → model", "边", "工具结果反馈给 LLM（ReAct 循环）")
    console.print(table)

    # 示例 1：简单对话（不触发工具）
    console.print("\n[bold yellow]示例 1：简单对话（无工具调用）[/bold yellow]")
    result = agent.invoke({"messages": [HumanMessage(content="你好，介绍一下你自己")]})
    console.print(f"回答: {result['messages'][-1].content[:200]}")

    # 示例 2：触发工具调用
    console.print("\n[bold yellow]示例 2：数学计算（触发工具调用）[/bold yellow]")
    result = agent.invoke({"messages": [HumanMessage(content="帮我算一下 (17 * 23) + (45 * 12) 等于多少")]})
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            console.print(f"  [dim]LLM 决定调用工具: {[c['name'] for c in msg.tool_calls]}[/dim]")
        elif isinstance(msg, ToolMessage):
            console.print(f"  [dim]工具返回: {msg.content}[/dim]")
    console.print(f"最终回答: {result['messages'][-1].content}")

    # 示例 3：多工具调用
    console.print("\n[bold yellow]示例 3：多步推理（多次工具调用）[/bold yellow]")
    result = agent.invoke({
        "messages": [HumanMessage(content="先算 2 的 10 次方，再告诉我结果的字符长度")]
    })
    console.print(f"最终回答: {result['messages'][-1].content}")

    # 设计对比总结
    console.print(Panel(
        "[bold]Phase 1 vs LangGraph 设计对比[/bold]\n\n"
        "Phase 1 ExecutionEngine:\n"
        "  while step < max_steps:\n"
        "      thought = think_fn(task, memory)\n"
        "      if is_final(thought): break\n"
        "      observation = act_fn(thought)\n"
        "      memory.append(observation)\n\n"
        "LangGraph StateGraph:\n"
        "  graph.add_node('model', call_model)\n"
        "  graph.add_node('tools', call_tools)\n"
        "  graph.add_conditional_edges('model', should_continue)\n"
        "  graph.add_edge('tools', 'model')\n\n"
        "本质相同（ReAct 循环），但图的表达方式带来：\n"
        "  1. 可视化 — 图可以直接画出来\n"
        "  2. 可组合 — 加节点加边，不改循环\n"
        "  3. 可持久化 — 每个节点间可以存检查点",
        title="设计思考",
        border_style="green",
    ))


if __name__ == "__main__":
    run_demo()
