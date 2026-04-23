"""
02_text_chunking.py — 文本分块策略

学习目标：
1. 理解为什么需要分块：LLM 上下文有限 + 检索精度需要小粒度
2. 掌握四种分块策略：固定大小、递归字符、语义分块、按文档结构
3. 理解 chunk_size 和 chunk_overlap 的权衡
4. 学会评估分块质量

核心概念：
- chunk_size：每个块的目标大小（字符数或 token 数）
- chunk_overlap：相邻块的重叠部分，防止语义被截断
- 分块粒度权衡：块太大 → 检索不精确；块太小 → 丢失上下文

分块策略对比：
    ┌─────────────────┬──────────┬──────────┬──────────┐
    │     策略         │  实现难度 │  语义保持 │  适用场景  │
    ├─────────────────┼──────────┼──────────┼──────────┤
    │ 固定大小         │   ★☆☆   │   ★☆☆   │ 快速原型   │
    │ 递归字符         │   ★★☆   │   ★★☆   │ 通用文档   │
    │ 语义分块         │   ★★★   │   ★★★   │ 高质量RAG  │
    │ 文档结构         │   ★★☆   │   ★★★   │ Markdown等 │
    └─────────────────┴──────────┴──────────┴──────────┘

运行方式：
    python 02_text_chunking.py
"""

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)


# ============================================================
# 1. 固定大小分块（最简单，但最粗暴）
# ============================================================

def chunk_by_fixed_size(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    按固定字符数切分。
    问题：可能在句子中间截断，破坏语义。
    优点：实现简单，速度快。
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# ============================================================
# 2. 递归字符分块（最常用的通用方案）
# ============================================================

def chunk_by_recursive_split(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """
    递归字符分块的核心思想：
    1. 先尝试用最大的分隔符（如 \\n\\n）切分
    2. 如果某个块还是太大，用下一级分隔符（如 \\n）继续切
    3. 最后兜底用字符级切分

    这就是 LangChain RecursiveCharacterTextSplitter 的核心算法。
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", ".", " ", ""]

    chunks: list[str] = []

    def _split(text: str, sep_idx: int) -> None:
        if len(text) <= chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return

        if sep_idx >= len(separators):
            chunks.append(text[:chunk_size].strip())
            remaining = text[chunk_size - chunk_overlap:]
            if remaining.strip():
                _split(remaining, sep_idx)
            return

        sep = separators[sep_idx]
        if not sep:
            chunks.append(text[:chunk_size].strip())
            remaining = text[chunk_size - chunk_overlap:]
            if remaining.strip():
                _split(remaining, 0)
            return

        parts = text.split(sep)
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    _split(part, sep_idx + 1)
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

    _split(text, 0)
    return chunks


# ============================================================
# 3. 按 Markdown 结构分块
# ============================================================

def chunk_by_markdown_headers(text: str) -> list[Document]:
    """
    按 Markdown 标题切分，保留层级结构作为元数据。
    适合结构化文档（技术文档、Wiki、README）。
    """
    import re

    chunks: list[Document] = []
    current_headers: dict[str, str] = {}
    current_content: list[str] = []

    for line in text.split("\n"):
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if header_match:
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    chunks.append(Document(
                        content=content,
                        metadata=dict(current_headers),
                    ))
                current_content = []

            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_headers[f"h{level}"] = title

            for i in range(level + 1, 7):
                current_headers.pop(f"h{i}", None)

            current_content.append(line)
        else:
            current_content.append(line)

    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            chunks.append(Document(
                content=content,
                metadata=dict(current_headers),
            ))

    return chunks


# ============================================================
# 4. 使用 LangChain 的分块器（生产推荐）
# ============================================================

def chunk_with_langchain(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    LangChain 的 RecursiveCharacterTextSplitter 是业界最常用的分块器。
    它的实现和我们上面手写的递归分块思路一致，但更健壮。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


# ============================================================
# 5. 分块质量评估
# ============================================================

def evaluate_chunks(chunks: list[str], original_text: str) -> dict:
    """评估分块质量的几个关键指标"""
    sizes = [len(c) for c in chunks]
    return {
        "总块数": len(chunks),
        "平均大小": round(sum(sizes) / len(sizes)) if sizes else 0,
        "最小块": min(sizes) if sizes else 0,
        "最大块": max(sizes) if sizes else 0,
        "大小标准差": round((sum((s - sum(sizes)/len(sizes))**2 for s in sizes) / len(sizes)) ** 0.5) if sizes else 0,
        "覆盖率": f"{sum(sizes) / len(original_text) * 100:.1f}%",
    }


# ============================================================
# 6. 演示
# ============================================================

def load_sample_text() -> str:
    sample_path = Path("sample_data/ai_agent_overview.txt")
    if not sample_path.exists():
        from importlib import import_module
        mod = import_module("01_document_loading")
        mod.create_sample_documents()
    return sample_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    console.print(Panel("✂️  文本分块策略对比", style="bold blue"))

    text = load_sample_text()
    console.print(f"原始文本长度: {len(text)} 字符\n")

    strategies = {
        "固定大小 (200字符)": chunk_by_fixed_size(text, 200, 50),
        "递归字符 (200字符)": chunk_by_recursive_split(text, 200, 50),
        "LangChain递归 (200字符)": chunk_with_langchain(text, 200, 50),
    }

    for name, chunks in strategies.items():
        stats = evaluate_chunks(chunks, text)
        table = Table(title=name, show_header=True)
        table.add_column("指标", style="cyan")
        table.add_column("值", style="white")
        for k, v in stats.items():
            table.add_row(k, str(v))
        console.print(table)
        console.print(f"  前两个块预览:")
        for i, chunk in enumerate(chunks[:2]):
            preview = chunk[:100].replace("\n", "↵")
            console.print(f"    [{i}] {preview}...")
        console.print()

    # Markdown 结构分块
    console.print("[bold]Markdown 结构分块：[/bold]")
    md_chunks = chunk_by_markdown_headers(text)
    for i, doc in enumerate(md_chunks[:3]):
        console.print(f"  块 {i}: headers={doc.metadata}, 长度={len(doc.content)}")

    console.print("\n[dim]下一步 → 03_embedding_vectorstore.py 学习向量化与向量数据库[/dim]")
