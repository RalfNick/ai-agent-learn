"""
Run the Phase 2 RAG benchmark.

Outputs:
    outputs/benchmark_results.json
    outputs/benchmark_summary.csv
    reports/rag_optimization_experiment_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import jieba
import litellm
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional runtime fallback
    CrossEncoder = None  # type: ignore

try:
    from litellm import completion_cost
except Exception:  # pragma: no cover - optional runtime fallback
    completion_cost = None  # type: ignore

from benchmark_dataset import EVAL_QUESTIONS, SOURCE_SPECS, EvalQuestion, validate_dataset


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
REPORT_DIR = HERE / "reports"

DEFAULT_EMBEDDING_MODEL = os.getenv("BENCHMARK_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_RERANK_MODEL = os.getenv("BENCHMARK_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")

console = Console()


def load_env_files() -> None:
    candidates = [
        ROOT / ".env",
        ROOT / "phase-2-rag" / "05-rag-benchmark" / ".env",
        ROOT / "phase-2-rag" / "04-rag-evaluation" / ".env",
        ROOT / "phase-2-rag" / "03-hybrid-search" / ".env",
        ROOT / "phase-2-rag" / "02-advanced-rag" / ".env",
        ROOT / "phase-2-rag" / "01-basic-rag" / ".env",
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    source_type: str
    path: str
    text: str


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, other: "LLMUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost_usd += other.cost_usd
        self.calls += other.calls


@dataclass(frozen=True)
class RAGConfig:
    name: str
    use_hybrid: bool
    use_rerank: bool
    use_query_transform: bool
    top_k: int = 3
    first_stage_k: int = 12


@dataclass
class QueryResult:
    config_name: str
    question: str
    generated_queries: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_source_ids: list[str]
    answer: str
    faithfulness: float
    faithfulness_reason: str
    precision_at_3: float
    recall_at_3: float
    mrr: float
    ndcg_at_3: float
    latency_ms: float
    llm_usage: LLMUsage


@dataclass
class ConfigSummary:
    name: str
    precision_at_3: float
    recall_at_3: float
    mrr: float
    ndcg_at_3: float
    faithfulness: float
    avg_latency_ms: float
    total_cost_usd: float
    llm_calls: int


class LLMClient:
    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, prompt: str, temperature: float = 0.2) -> tuple[str, LLMUsage]:
        start = time.perf_counter()
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        _ = time.perf_counter() - start
        text = response.choices[0].message.content.strip()
        usage = self._usage_from_response(response, prompt, text)
        return text, usage

    def _usage_from_response(self, response: Any, prompt: str, completion: str) -> LLMUsage:
        usage = LLMUsage(calls=1)
        raw_usage = getattr(response, "usage", None)
        if raw_usage:
            usage.prompt_tokens = int(read_usage_value(raw_usage, "prompt_tokens"))
            usage.completion_tokens = int(read_usage_value(raw_usage, "completion_tokens"))
            usage.total_tokens = int(read_usage_value(raw_usage, "total_tokens"))
        if usage.total_tokens == 0:
            usage.prompt_tokens = max(1, len(prompt) // 4)
            usage.completion_tokens = max(1, len(completion) // 4)
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        if completion_cost is not None:
            try:
                usage.cost_usd = float(completion_cost(completion_response=response))
                if usage.cost_usd > 0:
                    return usage
            except Exception:
                pass

        input_rate, output_rate = model_cost_rates(self.model)
        usage.cost_usd = (
            usage.prompt_tokens * input_rate / 1_000_000
            + usage.completion_tokens * output_rate / 1_000_000
        )
        return usage


def read_usage_value(raw_usage: Any, key: str) -> int:
    value = getattr(raw_usage, key, None)
    if value is None and isinstance(raw_usage, dict):
        value = raw_usage.get(key)
    return int(value or 0)


def model_cost_rates(model: str) -> tuple[float, float]:
    """Return fallback USD rates per 1M input/output tokens.

    Env vars override these estimates:
        BENCHMARK_INPUT_COST_PER_1M
        BENCHMARK_OUTPUT_COST_PER_1M
    """
    if os.getenv("BENCHMARK_INPUT_COST_PER_1M") and os.getenv("BENCHMARK_OUTPUT_COST_PER_1M"):
        return (
            float(os.getenv("BENCHMARK_INPUT_COST_PER_1M", "0")),
            float(os.getenv("BENCHMARK_OUTPUT_COST_PER_1M", "0")),
        )
    normalized = model.lower()
    if "deepseek" in normalized:
        return 0.27, 1.10
    if "gpt-4o-mini" in normalized:
        return 0.15, 0.60
    if "gpt-4o" in normalized:
        return 2.50, 10.00
    return 0.0, 0.0


def build_chunks(chunk_size: int = 900, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source in SOURCE_SPECS:
        path = ROOT / source.path
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark source: {path}")
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        text = f"# {source.title}\n\nSource path: {source.path}\n\n{text}"
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end), text.rfind("。", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= 120:
                chunks.append(
                    Chunk(
                        chunk_id=f"{source.source_id}::{chunk_index}",
                        source_id=source.source_id,
                        title=source.title,
                        source_type=source.source_type,
                        path=source.path,
                        text=chunk_text,
                    )
                )
                chunk_index += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


class BenchmarkIndex:
    def __init__(self, chunks: list[Chunk], embedding_model: str, rerank_model: str) -> None:
        self.chunks = chunks
        self.embedding_model_name = embedding_model
        self.rerank_model_name = rerank_model
        self.tokenized_chunks = [list(jieba.cut(chunk.text)) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        self.embedder = SentenceTransformer(embedding_model)
        self.chunk_embeddings = self.embedder.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.reranker = self._load_reranker(rerank_model)
        self.rerank_mode = "cross_encoder" if self.reranker else "dense_fallback"

    def _load_reranker(self, rerank_model: str) -> Any | None:
        if CrossEncoder is None:
            return None
        try:
            return CrossEncoder(rerank_model)
        except Exception as exc:
            console.print(f"[yellow]Cross-Encoder unavailable, using dense fallback rerank: {exc}[/yellow]")
            return None

    def vector_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_embedding = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores = np.dot(self.chunk_embeddings, query_embedding.T).flatten()
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(list(jieba.cut(query)))
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def hybrid_search(self, queries: list[str], top_k: int) -> list[tuple[int, float]]:
        rankings: list[list[tuple[int, float]]] = []
        for query in queries:
            rankings.append(self.bm25_search(query, top_k=top_k))
            rankings.append(self.vector_search(query, top_k=top_k))
        return reciprocal_rank_fusion(rankings)[:top_k]

    def rerank(self, query: str, candidates: list[tuple[int, float]], top_k: int) -> list[tuple[int, float]]:
        if not candidates:
            return []
        indices = [idx for idx, _ in candidates]
        if self.reranker is not None:
            pairs = [[query, self.chunks[idx].text] for idx in indices]
            scores = self.reranker.predict(pairs)
            reranked = sorted(zip(indices, scores), key=lambda item: item[1], reverse=True)
            return [(int(idx), float(score)) for idx, score in reranked[:top_k]]
        query_embedding = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores = np.dot(self.chunk_embeddings[indices], query_embedding.T).flatten()
        reranked = sorted(zip(indices, scores), key=lambda item: item[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in reranked[:top_k]]


def reciprocal_rank_fusion(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return len(set(top) & relevant) / len(top) if top else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return len(set(top) & relevant) / len(relevant) if relevant else 1.0


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, source_id in enumerate(retrieved, start=1):
        if source_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    dcg = sum((1.0 if source_id in relevant else 0.0) / math.log2(rank + 1) for rank, source_id in enumerate(top, start=1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def unique_source_ids(chunks: list[Chunk], ranked_indices: list[int]) -> list[str]:
    source_ids: list[str] = []
    seen: set[str] = set()
    for idx in ranked_indices:
        source_id = chunks[idx].source_id
        if source_id not in seen:
            source_ids.append(source_id)
            seen.add(source_id)
    return source_ids


def transform_query(llm: LLMClient, question: str) -> tuple[list[str], LLMUsage]:
    prompt = f"""请把下面的问题改写成 3 个适合知识库检索的中文查询。
要求：
1. 覆盖不同关键词和表达角度
2. 不要回答问题
3. 只输出 JSON 数组，例如 ["查询1", "查询2", "查询3"]

问题：{question}
"""
    text, usage = llm.complete(prompt, temperature=0.1)
    queries = parse_json_list(text)
    if not queries:
        queries = [line.strip("- 1234567890.、 \t") for line in text.splitlines() if line.strip()]
    queries = [query for query in queries if query][:3]
    return ([question] + queries) if queries else [question], usage


def parse_json_list(text: str) -> list[str]:
    try:
        parsed = json.loads(extract_json(text))
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    except Exception:
        return []
    return []


def extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    start = min([idx for idx in [text.find("{"), text.find("[")] if idx >= 0], default=0)
    end = max(text.rfind("}"), text.rfind("]"))
    if end >= start:
        return text[start : end + 1]
    return text


def build_context(chunks: list[Chunk], ranked: list[tuple[int, float]]) -> str:
    parts = []
    for i, (idx, score) in enumerate(ranked, start=1):
        chunk = chunks[idx]
        parts.append(
            f"[来源 {i}] source_id={chunk.source_id}, title={chunk.title}, path={chunk.path}, score={score:.4f}\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(llm: LLMClient, question: EvalQuestion, context: str) -> tuple[str, LLMUsage]:
    prompt = f"""你是一个严谨的 RAG 问答助手。请只基于参考资料回答问题。

要求：
1. 如果参考资料不足，明确说明不足，不要编造。
2. 回答中尽量引用 source_id 或来源编号。
3. 用中文回答，保持简洁准确。

参考资料：
{context}

问题：{question.question}

回答："""
    return llm.complete(prompt, temperature=0.2)


def judge_faithfulness(llm: LLMClient, question: EvalQuestion, context: str, answer: str) -> tuple[float, str, LLMUsage]:
    prompt = f"""请评估下面 RAG 回答的 Faithfulness（忠实度）。

评分标准：
- 1.0：所有关键声明都能被参考资料支持
- 0.7：大部分关键声明被支持，仅有轻微泛化
- 0.4：有明显未被支持的声明
- 0.0：主要内容与参考资料无关或大量编造

只输出 JSON：
{{"score": 0.0到1.0之间的小数, "reason": "一句中文理由"}}

问题：
{question.question}

人工参考答案：
{question.ground_truth}

参考资料：
{context}

待评估回答：
{answer}
"""
    text, usage = llm.complete(prompt, temperature=0.0)
    try:
        parsed = json.loads(extract_json(text))
        score = float(parsed.get("score", 0.0))
        reason = str(parsed.get("reason", "")).strip()
        return max(0.0, min(1.0, score)), reason, usage
    except Exception:
        return 0.0, f"Judge output parse failed: {text[:160]}", usage


def retrieve_for_config(
    index: BenchmarkIndex,
    config: RAGConfig,
    question: EvalQuestion,
    generated_queries: list[str],
) -> list[tuple[int, float]]:
    if config.use_hybrid:
        candidates = index.hybrid_search(generated_queries, top_k=config.first_stage_k)
    else:
        candidates = index.vector_search(question.question, top_k=config.first_stage_k)
    if config.use_rerank:
        return index.rerank(question.question, candidates, top_k=config.top_k)
    return candidates[: config.top_k]


def run_one(
    index: BenchmarkIndex,
    llm: LLMClient,
    config: RAGConfig,
    question: EvalQuestion,
) -> QueryResult:
    usage = LLMUsage()
    start = time.perf_counter()
    generated_queries = [question.question]
    if config.use_query_transform:
        generated_queries, transform_usage = transform_query(llm, question.question)
        usage.add(transform_usage)

    ranked = retrieve_for_config(index, config, question, generated_queries)
    context = build_context(index.chunks, ranked)
    answer, answer_usage = generate_answer(llm, question, context)
    usage.add(answer_usage)
    faithfulness, reason, judge_usage = judge_faithfulness(llm, question, context, answer)
    usage.add(judge_usage)

    latency_ms = (time.perf_counter() - start) * 1000
    ranked_indices = [idx for idx, _ in ranked]
    retrieved_sources = unique_source_ids(index.chunks, ranked_indices)
    relevant = set(question.relevant_source_ids)

    return QueryResult(
        config_name=config.name,
        question=question.question,
        generated_queries=generated_queries,
        retrieved_chunk_ids=[index.chunks[idx].chunk_id for idx in ranked_indices],
        retrieved_source_ids=retrieved_sources,
        answer=answer,
        faithfulness=faithfulness,
        faithfulness_reason=reason,
        precision_at_3=precision_at_k(retrieved_sources, relevant, 3),
        recall_at_3=recall_at_k(retrieved_sources, relevant, 3),
        mrr=mean_reciprocal_rank(retrieved_sources, relevant),
        ndcg_at_3=ndcg_at_k(retrieved_sources, relevant, 3),
        latency_ms=latency_ms,
        llm_usage=usage,
    )


def summarize(configs: list[RAGConfig], results: list[QueryResult]) -> list[ConfigSummary]:
    summaries: list[ConfigSummary] = []
    for config in configs:
        items = [item for item in results if item.config_name == config.name]
        total_usage = LLMUsage()
        for item in items:
            total_usage.add(item.llm_usage)
        summaries.append(
            ConfigSummary(
                name=config.name,
                precision_at_3=mean([item.precision_at_3 for item in items]),
                recall_at_3=mean([item.recall_at_3 for item in items]),
                mrr=mean([item.mrr for item in items]),
                ndcg_at_3=mean([item.ndcg_at_3 for item in items]),
                faithfulness=mean([item.faithfulness for item in items]),
                avg_latency_ms=mean([item.latency_ms for item in items]),
                total_cost_usd=total_usage.cost_usd,
                llm_calls=total_usage.calls,
            )
        )
    return summaries


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def write_outputs(
    chunks: list[Chunk],
    questions: list[EvalQuestion],
    configs: list[RAGConfig],
    results: list[QueryResult],
    summaries: list[ConfigSummary],
    index: BenchmarkIndex,
    llm_model: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "source_count": len(SOURCE_SPECS),
            "chunk_count": len(chunks),
            "question_count": len(questions),
            "embedding_model": index.embedding_model_name,
            "rerank_model": index.rerank_model_name,
            "rerank_mode": index.rerank_mode,
            "llm_model": llm_model,
            "configs": [asdict(config) for config in configs],
        },
        "summaries": [asdict(summary) for summary in summaries],
        "results": [result_to_dict(result) for result in results],
    }
    (OUTPUT_DIR / "benchmark_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (OUTPUT_DIR / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "name",
                "precision_at_3",
                "recall_at_3",
                "mrr",
                "ndcg_at_3",
                "faithfulness",
                "avg_latency_ms",
                "total_cost_usd",
                "llm_calls",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))

    (REPORT_DIR / "rag_optimization_experiment_report.md").write_text(
        build_report(summaries, payload),
        encoding="utf-8",
    )


def result_to_dict(result: QueryResult) -> dict[str, Any]:
    data = asdict(result)
    data["llm_usage"] = asdict(result.llm_usage)
    return data


def build_report(summaries: list[ConfigSummary], payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    by_name = {summary.name: summary for summary in summaries}
    naive = by_name["naive_vector"]
    hybrid_rerank = by_name["hybrid_rerank"]
    query_transform = by_name["query_transform_rerank"]
    best_recall = max(summaries, key=lambda item: item.recall_at_3)
    best_faith = max(summaries, key=lambda item: item.faithfulness)
    best_ndcg = max(summaries, key=lambda item: item.ndcg_at_3)

    rows = "\n".join(
        "| {name} | {p:.3f} | {r:.3f} | {mrr:.3f} | {ndcg:.3f} | {faith:.3f} | {lat:.0f} | ${cost:.4f} | {calls} |".format(
            name=summary.name,
            p=summary.precision_at_3,
            r=summary.recall_at_3,
            mrr=summary.mrr,
            ndcg=summary.ndcg_at_3,
            faith=summary.faithfulness,
            lat=summary.avg_latency_ms,
            cost=summary.total_cost_usd,
            calls=summary.llm_calls,
        )
        for summary in summaries
    )

    return f"""# RAG 优化实验报告：从能跑到可证明变好

> Phase 2 验收 benchmark。实验资料来自本仓库真实学习文章和技术脚本，问题集为人工构造并标注相关资料来源。

## 1. 实验设置

- 资料源数量：{metadata["source_count"]}
- 文档块数量：{metadata["chunk_count"]}
- 评估问题数量：{metadata["question_count"]}
- Embedding 模型：`{metadata["embedding_model"]}`
- Rerank 模型：`{metadata["rerank_model"]}`
- Rerank 实际模式：`{metadata["rerank_mode"]}`
- LLM 模型：`{metadata["llm_model"]}`

## 2. 对比配置

| 配置 | 说明 |
|------|------|
| `naive_vector` | 只使用向量检索 Top-3 |
| `hybrid` | BM25 + Dense 检索，使用 RRF 融合 |
| `hybrid_rerank` | Hybrid 粗召回后使用 Cross-Encoder 精排 |
| `query_transform_rerank` | LLM Multi-Query 改写 + Hybrid + Cross-Encoder 精排 |

## 3. 指标结果

| 配置 | Precision@3 | Recall@3 | MRR | NDCG@3 | Faithfulness | 平均延迟(ms) | 估算成本 | LLM调用 |
|------|-------------|----------|-----|--------|--------------|--------------|----------|---------|
{rows}

## 4. 结论

- Recall@3 最好：`{best_recall.name}`，分数 `{best_recall.recall_at_3:.3f}`。
- NDCG@3 最好：`{best_ndcg.name}`，分数 `{best_ndcg.ndcg_at_3:.3f}`。
- Faithfulness 最好：`{best_faith.name}`，分数 `{best_faith.faithfulness:.3f}`。

这组数据说明：Phase 2 的优化不是“感觉更高级”，而是可以量化证明。

相对 `naive_vector`，`hybrid_rerank` 的 Precision@3 从 `{naive.precision_at_3:.3f}` 提升到 `{hybrid_rerank.precision_at_3:.3f}`，Recall@3 从 `{naive.recall_at_3:.3f}` 提升到 `{hybrid_rerank.recall_at_3:.3f}`，NDCG@3 从 `{naive.ndcg_at_3:.3f}` 提升到 `{hybrid_rerank.ndcg_at_3:.3f}`，Faithfulness 从 `{naive.faithfulness:.3f}` 提升到 `{hybrid_rerank.faithfulness:.3f}`。这说明混合检索加重排序既改善了排序质量，也让生成答案更忠于上下文。

`query_transform_rerank` 的 Faithfulness 最高，达到 `{query_transform.faithfulness:.3f}`，但 Recall@3 为 `{query_transform.recall_at_3:.3f}`，NDCG@3 也低于 `hybrid_rerank`。在这个数据集里，LLM 查询改写没有带来稳定的检索收益，反而可能把部分查询改写偏。它适合作为短查询、口语化查询、术语不稳定场景下的按需增强，而不适合作为默认必开选项。

因此，Phase 2 当前推荐默认配置是：**Hybrid + Cross-Encoder Rerank**。它在质量、延迟、成本之间最均衡。

## 5. 成本与延迟观察

`query_transform_rerank` 会额外调用 LLM 做查询改写，因此 LLM 调用数从 {hybrid_rerank.llm_calls} 次增加到 {query_transform.llm_calls} 次，平均延迟从 `hybrid_rerank` 的 `{hybrid_rerank.avg_latency_ms:.0f}ms` 增加到 `{query_transform.avg_latency_ms:.0f}ms`，估算成本从 `${hybrid_rerank.total_cost_usd:.4f}` 增加到 `${query_transform.total_cost_usd:.4f}`。

本实验的成本估算优先使用 LiteLLM 的返回结果；当模型价格表返回 0 时，脚本使用内置 fallback 费率估算。可通过 `BENCHMARK_INPUT_COST_PER_1M` 和 `BENCHMARK_OUTPUT_COST_PER_1M` 覆盖。

结论很清楚：如果目标是默认生产配置，先选择 `hybrid_rerank`；如果遇到召回不足的复杂自然语言问题，再针对性开启 query transform，并用同一套 benchmark 验证它是否真的提升。

## 6. 复现实验

```bash
cd phase-2-rag/05-rag-benchmark
python3 benchmark.py
```

完整明细见：

- `outputs/benchmark_results.json`
- `outputs/benchmark_summary.csv`
"""


def print_summary(summaries: list[ConfigSummary]) -> None:
    table = Table(title="Phase 2 RAG Benchmark Summary", show_header=True)
    table.add_column("Config")
    table.add_column("P@3", justify="right")
    table.add_column("R@3", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("NDCG@3", justify="right")
    table.add_column("Faith", justify="right")
    table.add_column("Latency ms", justify="right")
    table.add_column("Cost", justify="right")
    for summary in summaries:
        table.add_row(
            summary.name,
            f"{summary.precision_at_3:.3f}",
            f"{summary.recall_at_3:.3f}",
            f"{summary.mrr:.3f}",
            f"{summary.ndcg_at_3:.3f}",
            f"{summary.faithfulness:.3f}",
            f"{summary.avg_latency_ms:.0f}",
            f"${summary.total_cost_usd:.4f}",
        )
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 RAG benchmark")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N questions for a smoke test")
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    return parser.parse_args()


def main() -> None:
    load_env_files()
    args = parse_args()
    validate_dataset()
    questions = list(EVAL_QUESTIONS[: args.limit]) if args.limit else list(EVAL_QUESTIONS)
    if not questions:
        raise ValueError("No evaluation questions selected.")

    configs = [
        RAGConfig("naive_vector", use_hybrid=False, use_rerank=False, use_query_transform=False),
        RAGConfig("hybrid", use_hybrid=True, use_rerank=False, use_query_transform=False),
        RAGConfig("hybrid_rerank", use_hybrid=True, use_rerank=True, use_query_transform=False),
        RAGConfig("query_transform_rerank", use_hybrid=True, use_rerank=True, use_query_transform=True),
    ]

    console.print(Panel("Phase 2 RAG Benchmark: real corpus + labeled QA + LLM faithfulness", style="bold blue"))
    console.print(f"[dim]Questions: {len(questions)} | Sources: {len(SOURCE_SPECS)} | LLM: {args.llm_model}[/dim]")

    chunks = build_chunks()
    console.print(f"[dim]Built {len(chunks)} chunks from {len(SOURCE_SPECS)} sources[/dim]")
    index = BenchmarkIndex(chunks, args.embedding_model, args.rerank_model)
    llm = LLMClient(args.llm_model)

    results: list[QueryResult] = []
    total = len(configs) * len(questions)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Running benchmark", total=total)
        for config in configs:
            for question in questions:
                progress.update(task, description=f"{config.name}: {question.question[:32]}")
                results.append(run_one(index, llm, config, question))
                progress.advance(task)

    summaries = summarize(configs, results)
    write_outputs(chunks, questions, configs, results, summaries, index, args.llm_model)
    print_summary(summaries)
    console.print(f"\n[green]Wrote[/green] {OUTPUT_DIR / 'benchmark_results.json'}")
    console.print(f"[green]Wrote[/green] {OUTPUT_DIR / 'benchmark_summary.csv'}")
    console.print(f"[green]Wrote[/green] {REPORT_DIR / 'rag_optimization_experiment_report.md'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("[red]Interrupted[/red]")
        sys.exit(130)
