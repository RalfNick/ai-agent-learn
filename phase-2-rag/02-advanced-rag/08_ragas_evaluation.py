"""
08_ragas_evaluation.py — RAGAS 评估框架

学习目标：
1. 理解为什么需要系统化评估 RAG（不能只靠"看起来对"）
2. 掌握 RAGAS 的四个核心指标
3. 学会构建评估数据集
4. 实现手动版评估指标，理解底层原理

核心指标：
    ┌─────────────────────────────────────────────────────────┐
    │                    RAGAS 四大指标                         │
    │                                                         │
    │  ┌──────────────────┐    ┌──────────────────┐           │
    │  │ Faithfulness     │    │ Answer Relevancy │           │
    │  │ 忠实度            │    │ 答案相关性        │           │
    │  │ 回答是否基于上下文 │    │ 回答是否切题      │           │
    │  └──────────────────┘    └──────────────────┘           │
    │                                                         │
    │  ┌──────────────────┐    ┌──────────────────┐           │
    │  │ Context Precision│    │ Context Recall   │           │
    │  │ 上下文精确度      │    │ 上下文召回率      │           │
    │  │ 检索的文档是否相关│    │ 是否检索到所有     │           │
    │  │                  │    │ 需要的信息        │           │
    │  └──────────────────┘    └──────────────────┘           │
    └─────────────────────────────────────────────────────────┘

运行方式：
    python 08_ragas_evaluation.py
"""

import os

import litellm
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()

LLM_MODEL = "deepseek/deepseek-chat"


def call_llm(prompt: str, temperature: float = 0.1) -> str:
    response = litellm.completion(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# 1. 评估数据集
# ============================================================

EVAL_DATASET = [
    {
        "question": "RAG 的核心流程是什么？",
        "contexts": [
            "RAG（检索增强生成）的核心流程包括四个步骤：文档分块、向量化、检索和生成。首先将文档切分成小块，然后将每个块转换为向量存入向量数据库，用户提问时检索最相关的块，最后将这些块作为上下文让 LLM 生成回答。",
            "RAG 解决了 LLM 的两大痛点：知识过时和幻觉问题。通过检索外部知识库，LLM 可以基于最新的、真实的信息来回答问题。",
        ],
        "answer": "RAG 的核心流程包括四个步骤：1) 文档分块（Chunking）：将文档切分成小块；2) 向量化（Embedding）：将每个块转换为向量；3) 检索（Retrieval）：用户提问时找到最相关的块；4) 生成（Generation）：将相关块作为上下文让 LLM 生成回答。",
        "ground_truth": "RAG 的核心流程是：分块、向量化、检索、生成。",
    },
    {
        "question": "BM25 和向量检索有什么区别？",
        "contexts": [
            "BM25 是经典的稀疏检索算法，基于词频-逆文档频率来计算相关性，擅长精确的关键词匹配。",
            "向量检索将文本编码为高维向量，通过余弦相似度衡量语义相关性，能捕获文本的深层语义关系。",
        ],
        "answer": "BM25 是基于关键词匹配的稀疏检索，通过词频和逆文档频率计算相关性，擅长精确匹配。向量检索是基于语义的稠密检索，将文本编码为向量后通过余弦相似度衡量相关性，擅长语义匹配。两者互补，混合使用效果最好。",
        "ground_truth": "BM25 基于关键词匹配（稀疏检索），向量检索基于语义相似度（稠密检索）。",
    },
    {
        "question": "什么是 Cross-Encoder 重排序？",
        "contexts": [
            "混合检索结合 BM25 和向量检索的优势，通过 RRF 融合排序。",
            "HNSW 是一种高效的近似最近邻搜索算法。",
        ],
        "answer": "Cross-Encoder 重排序是一种精细排序技术。它同时接收查询和文档作为输入，通过深度交互编码输出精确的相关性分数。与 Bi-Encoder 不同，Cross-Encoder 能捕获查询和文档之间的细粒度交互信息，因此精度更高，但计算成本也更大。",
        "ground_truth": "Cross-Encoder 同时编码查询和文档对，输出相关性分数，精度高但速度慢。",
    },
]


# ============================================================
# 2. 手动实现 Faithfulness（忠实度）
# ============================================================

def evaluate_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    忠实度：回答中的每个声明是否都能在上下文中找到依据？
    分数 = 有依据的声明数 / 总声明数

    这是 RAG 最重要的指标 — 如果回答不忠于检索到的文档，
    那 RAG 就失去了意义（还不如直接问 LLM）。
    """
    context_text = "\n".join(contexts)

    extract_prompt = f"""请从以下回答中提取所有独立的事实性声明。每行一个声明。

回答：{answer}

声明列表："""

    claims_text = call_llm(extract_prompt)
    claims = [c.strip().lstrip("0123456789.-) ") for c in claims_text.split("\n") if c.strip()]

    if not claims:
        return 1.0

    supported_count = 0
    for claim in claims:
        verify_prompt = f"""判断以下声明是否能从给定的上下文中得到支持。
只回答 "是" 或 "否"。

上下文：{context_text}

声明：{claim}

判断："""

        result = call_llm(verify_prompt)
        if "是" in result:
            supported_count += 1

    score = supported_count / len(claims)
    console.print(f"  [dim]声明数: {len(claims)}, 有依据: {supported_count}[/dim]")
    return score


# ============================================================
# 3. 手动实现 Answer Relevancy（答案相关性）
# ============================================================

def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """
    答案相关性：回答是否切题？
    方法：从回答反向生成问题，看生成的问题和原始问题是否相似。
    """
    gen_prompt = f"""基于以下回答，生成 3 个可能的原始问题。每行一个。

回答：{answer}

可能的问题："""

    generated_qs = call_llm(gen_prompt)
    gen_questions = [q.strip().lstrip("0123456789.-) ") for q in generated_qs.split("\n") if q.strip()][:3]

    if not gen_questions:
        return 0.0

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([question], normalize_embeddings=True)
    gen_embs = model.encode(gen_questions, normalize_embeddings=True)

    similarities = np.dot(gen_embs, q_emb.T).flatten()
    score = float(np.mean(similarities))

    console.print(f"  [dim]生成问题数: {len(gen_questions)}, 平均相似度: {score:.3f}[/dim]")
    return max(0.0, min(1.0, score))


# ============================================================
# 4. 手动实现 Context Precision（上下文精确度）
# ============================================================

def evaluate_context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    """
    上下文精确度：检索到的文档中，有多少是真正相关的？
    高精确度 = 检索结果中噪声少
    """
    relevant_count = 0
    for ctx in contexts:
        judge_prompt = f"""判断以下上下文是否与回答问题相关。只回答 "是" 或 "否"。

问题：{question}
参考答案：{ground_truth}
上下文：{ctx}

是否相关："""

        result = call_llm(judge_prompt)
        if "是" in result:
            relevant_count += 1

    score = relevant_count / len(contexts) if contexts else 0.0
    console.print(f"  [dim]上下文数: {len(contexts)}, 相关: {relevant_count}[/dim]")
    return score


# ============================================================
# 5. 手动实现 Context Recall（上下文召回率）
# ============================================================

def evaluate_context_recall(contexts: list[str], ground_truth: str) -> float:
    """
    上下文召回率：ground truth 中的信息是否都被检索到了？
    低召回率 = 遗漏了关键信息
    """
    context_text = "\n".join(contexts)

    extract_prompt = f"""请从以下参考答案中提取所有关键信息点。每行一个。

参考答案：{ground_truth}

关键信息点："""

    points_text = call_llm(extract_prompt)
    points = [p.strip().lstrip("0123456789.-) ") for p in points_text.split("\n") if p.strip()]

    if not points:
        return 1.0

    found_count = 0
    for point in points:
        check_prompt = f"""判断以下信息点是否能在给定的上下文中找到。只回答 "是" 或 "否"。

上下文：{context_text}

信息点：{point}

是否找到："""

        result = call_llm(check_prompt)
        if "是" in result:
            found_count += 1

    score = found_count / len(points)
    console.print(f"  [dim]信息点数: {len(points)}, 已覆盖: {found_count}[/dim]")
    return score


# ============================================================
# 6. 综合评估
# ============================================================

def evaluate_rag_system(dataset: list[dict]) -> dict:
    """对整个数据集运行评估"""
    all_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for i, sample in enumerate(dataset):
        console.print(f"\n[bold]评估样本 {i+1}/{len(dataset)}：{sample['question']}[/bold]")

        console.print("  计算 Faithfulness...")
        f_score = evaluate_faithfulness(sample["answer"], sample["contexts"])
        all_scores["faithfulness"].append(f_score)

        console.print("  计算 Answer Relevancy...")
        ar_score = evaluate_answer_relevancy(sample["question"], sample["answer"])
        all_scores["answer_relevancy"].append(ar_score)

        console.print("  计算 Context Precision...")
        cp_score = evaluate_context_precision(sample["question"], sample["contexts"], sample["ground_truth"])
        all_scores["context_precision"].append(cp_score)

        console.print("  计算 Context Recall...")
        cr_score = evaluate_context_recall(sample["contexts"], sample["ground_truth"])
        all_scores["context_recall"].append(cr_score)

    avg_scores = {k: sum(v) / len(v) for k, v in all_scores.items()}
    return avg_scores


# ============================================================
# 7. 演示
# ============================================================

if __name__ == "__main__":
    console.print(Panel("📊 RAGAS 评估框架", style="bold blue"))

    console.print("[bold]手动实现 RAGAS 四大指标[/bold]")
    console.print("[dim]（使用 LLM 作为评估器，理解每个指标的计算原理）[/dim]\n")

    scores = evaluate_rag_system(EVAL_DATASET)

    # 结果汇总
    console.print()
    table = Table(title="📊 RAG 系统评估结果", show_header=True)
    table.add_column("指标", style="cyan")
    table.add_column("中文名", style="white")
    table.add_column("分数", style="green")
    table.add_column("含义", style="dim")

    rows = [
        ("Faithfulness", "忠实度", scores["faithfulness"], "回答是否基于检索到的文档"),
        ("Answer Relevancy", "答案相关性", scores["answer_relevancy"], "回答是否切题"),
        ("Context Precision", "上下文精确度", scores["context_precision"], "检索结果中相关文档的比例"),
        ("Context Recall", "上下文召回率", scores["context_recall"], "是否检索到了所有需要的信息"),
    ]

    for name, cn_name, score, desc in rows:
        color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
        table.add_row(name, cn_name, f"[{color}]{score:.3f}[/{color}]", desc)

    console.print(table)

    console.print("\n[bold]评估解读：[/bold]")
    console.print("  • Faithfulness < 0.7 → 回答可能包含幻觉，需要改进检索或 Prompt")
    console.print("  • Answer Relevancy < 0.7 → 回答跑题，需要改进 Prompt 模板")
    console.print("  • Context Precision < 0.7 → 检索噪声大，需要重排序或改进分块")
    console.print("  • Context Recall < 0.7 → 遗漏关键信息，需要改进检索策略")

    console.print("\n[dim]Phase 2 完成！你已经掌握了 RAG 管道的核心技术。[/dim]")
    console.print("[dim]下一步 → Phase 3 学习 LangChain、LangGraph 等框架[/dim]")
