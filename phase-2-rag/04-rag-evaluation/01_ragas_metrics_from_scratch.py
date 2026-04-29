"""
01_ragas_metrics_from_scratch.py — 手动实现 RAGAS 四大指标

学习目标：
1. 深入理解 RAGAS 每个指标的计算原理（不依赖 ragas 库）
2. 掌握 LLM-as-Judge 的评估范式
3. 学会构建评估数据集
4. 理解每个指标对 RAG 系统优化的指导意义

四大指标：
    ┌───────────────────────────────────────────────┐
    │              RAGAS 评估框架                      │
    │                                               │
    │  生成质量                                      │
    │  ├─ Faithfulness     回答是否忠于检索上下文      │
    │  └─ Answer Relevancy 回答是否切题              │
    │                                               │
    │  检索质量                                      │
    │  ├─ Context Precision 检索结果中相关文档比例     │
    │  └─ Context Recall    是否检索到所有需要的信息   │
    └───────────────────────────────────────────────┘

运行方式：
    cp .env.example .env  # 填入 API key
    python 01_ragas_metrics_from_scratch.py
"""

from __future__ import annotations

import os

import litellm
import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer

load_dotenv()

console = Console()

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")


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
        "answer": "RAG 的核心流程包括四个步骤：1) 文档分块：将文档切分成小块；2) 向量化：将每个块转换为向量；3) 检索：用户提问时找到最相关的块；4) 生成：将相关块作为上下文让 LLM 生成回答。",
        "ground_truth": "RAG 的核心流程是：分块、向量化、检索、生成。",
    },
    {
        "question": "BM25 和向量检索有什么区别？",
        "contexts": [
            "BM25 是经典的稀疏检索算法，基于词频-逆文档频率来计算相关性，擅长精确的关键词匹配。",
            "向量检索将文本编码为高维向量，通过余弦相似度衡量语义相关性，能捕获文本的深层语义关系。",
        ],
        "answer": "BM25 是基于关键词匹配的稀疏检索，通过词频和逆文档频率计算相关性。向量检索是基于语义的稠密检索，将文本编码为向量后通过余弦相似度衡量相关性。两者互补，混合使用效果最好。",
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
    {
        "question": "RAGAS 框架有哪些评估指标？",
        "contexts": [
            "RAGAS 是一个 RAG 系统评估框架，提供了忠实度（Faithfulness）、答案相关性（Answer Relevancy）、上下文精确度（Context Precision）和上下文召回率（Context Recall）四个核心指标。",
            "忠实度衡量回答是否基于检索到的上下文，答案相关性衡量回答是否切题。",
        ],
        "answer": "RAGAS 框架提供四个核心评估指标：1) Faithfulness（忠实度）：回答是否基于检索到的文档；2) Answer Relevancy（答案相关性）：回答是否切题；3) Context Precision（上下文精确度）：检索结果中相关文档的比例；4) Context Recall（上下文召回率）：是否检索到了所有需要的信息。",
        "ground_truth": "RAGAS 有四个指标：Faithfulness、Answer Relevancy、Context Precision、Context Recall。",
    },
    {
        "question": "如何提升 RAG 系统的检索质量？",
        "contexts": [
            "查询改写技术通过重新表述用户问题来提升检索效果，常见方法包括 HyDE 和多查询扩展。",
            "文本分块策略直接影响 RAG 系统的检索质量。常见方法包括固定大小分块、递归字符分块和基于语义的分块。",
        ],
        "answer": "提升 RAG 检索质量的方法包括：使用混合检索（BM25 + 向量）、添加 Cross-Encoder 重排序、优化分块策略、使用查询改写（HyDE、多查询扩展）。此外，选择合适的 Embedding 模型和调整 top_k 参数也很重要。",
        "ground_truth": "可以通过混合检索、重排序、查询改写、优化分块策略来提升检索质量。",
    },
]


# ============================================================
# 2. Faithfulness（忠实度）
# ============================================================

def evaluate_faithfulness(answer: str, contexts: list[str]) -> tuple[float, dict]:
    """
    忠实度：回答中的每个声明是否都能在上下文中找到依据？

    步骤：
    1. 从回答中提取所有事实性声明
    2. 逐一验证每个声明是否被上下文支持
    3. 分数 = 有依据的声明数 / 总声明数
    """
    context_text = "\n".join(contexts)

    extract_prompt = f"""请从以下回答中提取所有独立的事实性声明。
每行一个声明，不要编号，不要添加额外解释。

回答：{answer}

声明列表："""

    claims_text = call_llm(extract_prompt)
    claims = [c.strip().lstrip("0123456789.-) ") for c in claims_text.split("\n") if c.strip()]

    if not claims:
        return 1.0, {"claims": [], "supported": [], "total": 0}

    supported = []
    for claim in claims:
        verify_prompt = f"""判断以下声明是否能从给定的上下文中得到支持。
只回答 "是" 或 "否"。

上下文：{context_text}

声明：{claim}

判断："""
        result = call_llm(verify_prompt)
        is_supported = "是" in result
        supported.append(is_supported)

    score = sum(supported) / len(claims)
    details = {
        "claims": claims,
        "supported": supported,
        "total": len(claims),
        "supported_count": sum(supported),
    }
    return score, details


# ============================================================
# 3. Answer Relevancy（答案相关性）
# ============================================================

_embedding_model = None

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def evaluate_answer_relevancy(question: str, answer: str, n_generated: int = 3) -> tuple[float, dict]:
    """
    答案相关性：回答是否切题？

    方法（反向验证）：
    1. 从回答反向生成 N 个可能的原始问题
    2. 计算生成问题与真实问题的语义相似度
    3. 分数 = 平均相似度
    """
    gen_prompt = f"""基于以下回答，生成 {n_generated} 个可能的原始问题。
每行一个问题，不要编号。

回答：{answer}

可能的问题："""

    generated_qs = call_llm(gen_prompt)
    gen_questions = [
        q.strip().lstrip("0123456789.-) ")
        for q in generated_qs.split("\n")
        if q.strip()
    ][:n_generated]

    if not gen_questions:
        return 0.0, {"generated_questions": [], "similarities": []}

    model = _get_embedding_model()
    q_emb = model.encode([question], normalize_embeddings=True)
    gen_embs = model.encode(gen_questions, normalize_embeddings=True)

    similarities = np.dot(gen_embs, q_emb.T).flatten().tolist()
    score = float(np.mean(similarities))
    score = max(0.0, min(1.0, score))

    details = {
        "generated_questions": gen_questions,
        "similarities": [round(s, 4) for s in similarities],
    }
    return score, details


# ============================================================
# 4. Context Precision（上下文精确度）
# ============================================================

def evaluate_context_precision(
    question: str, contexts: list[str], ground_truth: str
) -> tuple[float, dict]:
    """
    上下文精确度：检索到的文档中，有多少是真正相关的？

    分数 = 相关文档数 / 检索文档总数
    """
    relevance = []
    for ctx in contexts:
        judge_prompt = f"""判断以下上下文是否与回答问题相关。只回答 "是" 或 "否"。

问题：{question}
参考答案：{ground_truth}
上下文：{ctx}

是否相关："""
        result = call_llm(judge_prompt)
        relevance.append("是" in result)

    score = sum(relevance) / len(contexts) if contexts else 0.0
    details = {
        "contexts_count": len(contexts),
        "relevant_count": sum(relevance),
        "relevance_per_ctx": relevance,
    }
    return score, details


# ============================================================
# 5. Context Recall（上下文召回率）
# ============================================================

def evaluate_context_recall(
    contexts: list[str], ground_truth: str
) -> tuple[float, dict]:
    """
    上下文召回率：ground truth 中的信息是否都被检索到了？

    步骤：
    1. 从 ground truth 提取关键信息点
    2. 检查每个信息点是否能在检索上下文中找到
    3. 分数 = 被覆盖的信息点数 / 总信息点数
    """
    context_text = "\n".join(contexts)

    extract_prompt = f"""请从以下参考答案中提取所有关键信息点。
每行一个信息点，不要编号。

参考答案：{ground_truth}

关键信息点："""

    points_text = call_llm(extract_prompt)
    points = [p.strip().lstrip("0123456789.-) ") for p in points_text.split("\n") if p.strip()]

    if not points:
        return 1.0, {"points": [], "found": [], "total": 0}

    found = []
    for point in points:
        check_prompt = f"""判断以下信息点是否能在给定的上下文中找到。只回答 "是" 或 "否"。

上下文：{context_text}

信息点：{point}

是否找到："""
        result = call_llm(check_prompt)
        found.append("是" in result)

    score = sum(found) / len(points)
    details = {
        "points": points,
        "found": found,
        "total": len(points),
        "found_count": sum(found),
    }
    return score, details


# ============================================================
# 6. 综合评估运行器
# ============================================================

def run_full_evaluation(dataset: list[dict]) -> dict[str, list[float]]:
    """对整个数据集运行四项评估"""
    all_scores: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for i, sample in enumerate(dataset):
        console.print(f"\n[bold]样本 {i+1}/{len(dataset)}：{sample['question']}[/bold]")

        console.print("  计算 Faithfulness...")
        f_score, f_detail = evaluate_faithfulness(sample["answer"], sample["contexts"])
        all_scores["faithfulness"].append(f_score)
        console.print(f"  [dim]声明数: {f_detail['total']}, 有依据: {f_detail['supported_count']} → {f_score:.2f}[/dim]")

        console.print("  计算 Answer Relevancy...")
        ar_score, ar_detail = evaluate_answer_relevancy(sample["question"], sample["answer"])
        all_scores["answer_relevancy"].append(ar_score)
        console.print(f"  [dim]生成问题: {len(ar_detail['generated_questions'])}, 相似度: {ar_detail['similarities']} → {ar_score:.2f}[/dim]")

        console.print("  计算 Context Precision...")
        cp_score, cp_detail = evaluate_context_precision(
            sample["question"], sample["contexts"], sample["ground_truth"]
        )
        all_scores["context_precision"].append(cp_score)
        console.print(f"  [dim]上下文: {cp_detail['contexts_count']}, 相关: {cp_detail['relevant_count']} → {cp_score:.2f}[/dim]")

        console.print("  计算 Context Recall...")
        cr_score, cr_detail = evaluate_context_recall(sample["contexts"], sample["ground_truth"])
        all_scores["context_recall"].append(cr_score)
        console.print(f"  [dim]信息点: {cr_detail['total']}, 覆盖: {cr_detail['found_count']} → {cr_score:.2f}[/dim]")

    return all_scores


def display_results(all_scores: dict[str, list[float]], dataset: list[dict]) -> None:
    """展示评估结果"""
    detail_table = Table(title="各样本评估分数", show_header=True)
    detail_table.add_column("问题", style="white", max_width=30)
    detail_table.add_column("Faith.", style="cyan", width=8)
    detail_table.add_column("Relev.", style="cyan", width=8)
    detail_table.add_column("Prec.", style="cyan", width=8)
    detail_table.add_column("Recall", style="cyan", width=8)

    for i, sample in enumerate(dataset):
        def _color(v: float) -> str:
            c = "green" if v >= 0.7 else "yellow" if v >= 0.5 else "red"
            return f"[{c}]{v:.2f}[/{c}]"

        detail_table.add_row(
            sample["question"][:28] + "...",
            _color(all_scores["faithfulness"][i]),
            _color(all_scores["answer_relevancy"][i]),
            _color(all_scores["context_precision"][i]),
            _color(all_scores["context_recall"][i]),
        )

    console.print(detail_table)

    summary_table = Table(title="评估汇总", show_header=True)
    summary_table.add_column("指标", style="cyan", width=20)
    summary_table.add_column("中文名", style="white", width=12)
    summary_table.add_column("平均分", style="green", width=8)
    summary_table.add_column("诊断建议", style="dim", max_width=40)

    metric_info = [
        ("Faithfulness", "忠实度", "faithfulness", "低 → 强化 Prompt，减少 temperature"),
        ("Answer Relevancy", "答案相关性", "answer_relevancy", "低 → 改进 Prompt 模板"),
        ("Context Precision", "上下文精确度", "context_precision", "低 → 加强重排序，减小 top_k"),
        ("Context Recall", "上下文召回率", "context_recall", "低 → 增大候选数，多查询扩展"),
    ]

    for name, cn, key, advice in metric_info:
        avg = sum(all_scores[key]) / len(all_scores[key])
        color = "green" if avg >= 0.7 else "yellow" if avg >= 0.5 else "red"
        summary_table.add_row(name, cn, f"[{color}]{avg:.3f}[/{color}]", advice)

    console.print(summary_table)


# ============================================================
# 7. 入口
# ============================================================

if __name__ == "__main__":
    console.print(Panel("RAGAS 四大指标 — 手动实现", style="bold blue"))
    console.print("[dim]使用 LLM-as-Judge 评估 RAG 系统质量[/dim]\n")

    all_scores = run_full_evaluation(EVAL_DATASET)
    console.print()
    display_results(all_scores, EVAL_DATASET)

    console.print("\n[dim]下一步 → 02_evaluation_pipeline.py 自动化评估不同 RAG 配置[/dim]")
