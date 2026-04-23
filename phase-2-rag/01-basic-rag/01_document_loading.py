"""
01_document_loading.py — 文档加载与预处理

学习目标：
1. 理解 RAG 的第一步：把非结构化数据变成可处理的文本
2. 掌握三种常见文档格式的加载：纯文本、PDF、网页
3. 理解文档元数据（metadata）的重要性
4. 学会基本的文本清洗技巧

核心概念：
- Document：一个文本块 + 它的元数据（来源、页码、标题等）
- Loader：负责从不同格式中提取文本
- Metadata：文档的"身份证"，后续检索时用于过滤和溯源

RAG 管道全景：
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ 文档加载  │───▶│ 文本分块  │───▶│ 向量化   │───▶│ 存入向量库│
    │ Loading  │    │ Chunking │    │Embedding │    │VectorDB  │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     ▲ 你在这里

运行方式：
    python 01_document_loading.py
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. 定义 Document 数据结构
# ============================================================

@dataclass
class Document:
    """一个文档块：文本内容 + 元数据"""
    content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Document(content='{preview}...', metadata={self.metadata})"


# ============================================================
# 2. 纯文本加载器
# ============================================================

def load_text_file(file_path: str) -> list[Document]:
    """加载纯文本文件"""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    return [Document(
        content=content,
        metadata={"source": str(path), "format": "text", "size": len(content)},
    )]


# ============================================================
# 3. PDF 加载器
# ============================================================

def load_pdf(file_path: str) -> list[Document]:
    """
    加载 PDF，每页生成一个 Document。
    为什么按页拆分？因为 PDF 的页码是天然的定位信息，
    后续检索时可以告诉用户"答案在第 X 页"。
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    documents: list[Document] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(Document(
                content=text,
                metadata={
                    "source": file_path,
                    "format": "pdf",
                    "page": i + 1,
                    "total_pages": len(reader.pages),
                },
            ))

    return documents


# ============================================================
# 4. 网页加载器
# ============================================================

def load_webpage(url: str) -> list[Document]:
    """
    加载网页内容，提取正文文本。
    实际生产中会用更复杂的提取器（如 trafilatura、readability），
    这里用 BeautifulSoup 演示核心思路。
    """
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url, timeout=10, headers={
        "User-Agent": "Mozilla/5.0 (Educational RAG Demo)"
    })
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    return [Document(
        content=text,
        metadata={"source": url, "format": "webpage", "title": soup.title.string if soup.title else ""},
    )]


# ============================================================
# 5. 文本清洗
# ============================================================

def clean_text(text: str) -> str:
    """
    基础文本清洗。RAG 的质量很大程度取决于输入文本的质量。
    垃圾进 → 垃圾出，这一步不能省。
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if len(line.strip()) > 1 or line.strip() == ""]
    return "\n".join(cleaned_lines).strip()


# ============================================================
# 6. 统一加载接口
# ============================================================

def load_document(source: str) -> list[Document]:
    """根据来源类型自动选择加载器"""
    if source.startswith(("http://", "https://")):
        docs = load_webpage(source)
    elif source.endswith(".pdf"):
        docs = load_pdf(source)
    else:
        docs = load_text_file(source)

    for doc in docs:
        doc.content = clean_text(doc.content)

    return docs


# ============================================================
# 7. 演示
# ============================================================

def create_sample_documents() -> str:
    """创建示例文档用于演示"""
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)

    sample_text = """# AI Agent 技术概览

## 什么是 AI Agent

AI Agent 是一种能够自主感知环境、做出决策并执行行动的智能系统。
与传统的 LLM 对话不同，Agent 具备以下核心能力：

1. 工具调用：Agent 可以调用外部工具（搜索引擎、计算器、API）来获取信息或执行操作。
2. 规划能力：Agent 能够将复杂任务分解为多个子步骤，并按顺序执行。
3. 记忆系统：Agent 可以记住之前的对话和操作结果，用于后续决策。
4. 自我反思：Agent 能够评估自己的输出质量，并在必要时调整策略。

## ReAct 框架

ReAct（Reasoning + Acting）是最经典的 Agent 框架之一。
它让 LLM 交替进行"思考"和"行动"：

- Thought：LLM 分析当前状态，决定下一步
- Action：调用工具执行操作
- Observation：观察工具返回的结果

这个循环持续进行，直到 Agent 认为任务完成。

## RAG（检索增强生成）

RAG 是让 LLM 基于外部知识回答问题的技术。核心流程：

1. 将文档切分成小块（Chunking）
2. 将每个块转换为向量（Embedding）
3. 用户提问时，找到最相关的块（Retrieval）
4. 将相关块作为上下文，让 LLM 生成回答（Generation）

RAG 解决了 LLM 的两大痛点：知识过时和幻觉问题。

## 多 Agent 系统

多个 Agent 协作完成复杂任务：
- Manager Agent：接收任务，分配给合适的 Worker
- Worker Agent：专注于特定领域（搜索、分析、编码等）
- 通信机制：Agent 之间通过消息传递协作

## 未来趋势

- MCP（Model Context Protocol）：统一的工具调用标准
- Agent 安全：防止 Prompt 注入和恶意工具调用
- 长期记忆：让 Agent 具备跨会话的持久记忆
"""

    sample_path = sample_dir / "ai_agent_overview.txt"
    sample_path.write_text(sample_text, encoding="utf-8")
    return str(sample_path)


if __name__ == "__main__":
    console.print(Panel("📄 文档加载与预处理", style="bold blue"))

    # 创建并加载示例文档
    sample_path = create_sample_documents()
    docs = load_document(sample_path)

    console.print(f"\n[green]✓ 加载了 {len(docs)} 个文档[/green]\n")

    for doc in docs:
        table = Table(title="文档信息", show_header=True)
        table.add_column("字段", style="cyan")
        table.add_column("值", style="white")

        table.add_row("来源", doc.metadata.get("source", ""))
        table.add_row("格式", doc.metadata.get("format", ""))
        table.add_row("内容长度", f"{len(doc.content)} 字符")
        table.add_row("内容预览", doc.content[:200] + "...")

        console.print(table)

    # 演示清洗效果
    console.print("\n[bold]文本清洗演示：[/bold]")
    dirty = "这是   一段    有很多\n\n\n\n\n空行和   多余空格的   文本\n\n\n结束"
    cleaned = clean_text(dirty)
    console.print(f"  清洗前: {repr(dirty)}")
    console.print(f"  清洗后: {repr(cleaned)}")

    console.print("\n[dim]下一步 → 02_text_chunking.py 学习文本分块策略[/dim]")
