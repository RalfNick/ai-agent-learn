"""
01_lcel_and_tools.py - LangChain 基础：LCEL、RunnableParallel 与 Tool schema

这个 demo 不依赖任何 LLM API key。

学习目标：
1. 用 LCEL 把 Prompt、Model、Parser 组合成一条 chain
2. 用 RunnableParallel 把同一个输入送入多个分支
3. 用 @tool 把普通函数转换成模型可理解的工具 schema

运行方式：
    pip install -r requirements.txt
    python3 01_lcel_and_tools.py
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def fake_chat_model(prompt_value: Any) -> str:
    """一个可预测的假模型，用来观察 LangChain 如何组织调用链。"""
    text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
    if "LangChain" in text:
        return "LangChain 的核心价值是把 Prompt、Model、Tool、Parser、Retriever 组合成可复用的 Runnable。"
    if "LangGraph" in text:
        return "LangGraph 的核心价值是把 Agent 工作流建模成显式状态图。"
    return f"收到输入：{text[:80]}"


def normalize_text(text: str) -> str:
    """清理空白并统一大小写，演示普通函数如何进入 Runnable。"""
    return " ".join(text.strip().split()).lower()


@tool
def word_count(text: str) -> int:
    """统计文本按空格切分后的词数。"""
    return len(text.split())


@tool
def contains_keyword(text: str, keyword: str) -> bool:
    """判断文本是否包含指定关键词。"""
    return keyword.lower() in text.lower()


def demo_lcel_chain() -> None:
    """Prompt -> Model -> Parser：LangChain 最基本的组合方式。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个 AI Agent 学习助手。回答要简洁。"),
            ("human", "用一句话解释 {topic} 的核心价值。"),
        ]
    )
    model = RunnableLambda(fake_chat_model)
    parser = StrOutputParser()

    chain = prompt | model | parser
    result = chain.invoke({"topic": "LangChain"})

    console.print(Panel(result, title="1. LCEL: Prompt | Model | Parser", border_style="cyan"))


def demo_parallel() -> None:
    """RunnableParallel：同一个输入并行进入多个分支。"""
    normalize = RunnableLambda(normalize_text)
    stats = RunnableParallel(
        {
            "raw": RunnablePassthrough(),
            "normalized": normalize,
            "length": normalize | RunnableLambda(len),
            "has_question_mark": normalize | RunnableLambda(lambda x: "?" in x or "？" in x),
        }
    )

    result = stats.invoke("  LangChain 和 LangGraph 有什么区别？  ")

    table = Table(title="2. RunnableParallel: 同一输入的多个视角")
    table.add_column("字段", style="cyan")
    table.add_column("值")
    for key, value in result.items():
        table.add_row(key, repr(value))
    console.print(table)


def demo_tools() -> None:
    """@tool：函数签名和 docstring 变成工具 schema。"""
    table = Table(title="3. Tool schema: 函数变成模型可调用的工具")
    table.add_column("工具", style="cyan")
    table.add_column("描述")
    table.add_column("参数 schema")

    for item in (word_count, contains_keyword):
        table.add_row(item.name, item.description or "", repr(item.args))
    console.print(table)

    console.print(
        Panel(
            "\n".join(
                [
                    f"word_count -> {word_count.invoke({'text': 'LangChain composes model tools parser'})}",
                    f"contains_keyword -> {contains_keyword.invoke({'text': 'Agent workflow needs state', 'keyword': 'state'})}",
                ]
            ),
            title="Tool invoke output",
            border_style="green",
        )
    )


def main() -> None:
    console.print(
        Panel(
            "[bold]LangChain Foundations[/bold]\n"
            "不调用真实 LLM，只观察框架抽象：LCEL、RunnableParallel、Tool schema。",
            title="00 LangChain Foundations",
            border_style="blue",
        )
    )
    demo_lcel_chain()
    demo_parallel()
    demo_tools()

    console.print(
        Panel(
            "这三个抽象会在 LangGraph 中继续出现：\n"
            "LangChain 负责把能力封装成可组合组件，LangGraph 负责把这些组件放进可控工作流。",
            title="Takeaway",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
