"""
LangGraph Agentic RAG workflow backed by the Phase 2 benchmark corpus.

This module turns the Phase 2 linear `hybrid_rerank` RAG pipeline into a graph:

    query_analysis -> retrieve -> context_grade
        relevant -> generate -> faithfulness_check -> end
        weak     -> rewrite -> retrieve
        failed   -> abstain

Faithfulness failures go through one repair pass before the graph either ends or
abstains. Every node appends to `route_trace` so benchmark reports can explain
which adaptive path the Agent took.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


ROOT = Path(__file__).resolve().parents[2]
PHASE2_BENCHMARK = ROOT / "phase-2-rag" / "05-rag-benchmark"
sys.path.insert(0, str(PHASE2_BENCHMARK))

from benchmark import (  # noqa: E402
    BenchmarkIndex,
    LLMClient,
    LLMUsage,
    build_chunks,
    build_context,
    judge_faithfulness,
    load_env_files,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    transform_query,
    unique_source_ids,
)
from benchmark_dataset import EvalQuestion  # noqa: E402


DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_LLM_MODEL = "deepseek/deepseek-chat"


class AgenticRAGState(TypedDict, total=False):
    question: str
    ground_truth: str
    relevant_source_ids: list[str]
    generated_queries: list[str]
    retrieved: list[dict[str, Any]]
    context: str
    context_score: float
    context_reason: str
    answer: str
    faithfulness: float
    faithfulness_reason: str
    retry_count: int
    repair_count: int
    abstained: bool
    route_trace: list[str]
    retrieval_metrics: dict[str, float]
    timings_ms: dict[str, float]
    llm_usage: dict[str, float]


@dataclass
class AgenticRAGResources:
    index: BenchmarkIndex
    llm: LLMClient
    top_k: int = 3
    first_stage_k: int = 12
    min_context_score: float = 0.62
    min_faithfulness: float = 0.86
    max_retries: int = 1
    max_repairs: int = 1


def build_resources(
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
) -> AgenticRAGResources:
    load_env_files()
    chunks = build_chunks()
    index = BenchmarkIndex(chunks, embedding_model, rerank_model)
    return AgenticRAGResources(index=index, llm=LLMClient(llm_model))


def build_agentic_rag_app(resources: AgenticRAGResources):
    graph = StateGraph(AgenticRAGState)

    graph.add_node("query_analysis", lambda state: query_analysis(state, resources))
    graph.add_node("retrieve", lambda state: retrieve(state, resources))
    graph.add_node("context_grade", lambda state: context_grade(state, resources))
    graph.add_node("query_rewrite", lambda state: query_rewrite(state, resources))
    graph.add_node("generate", lambda state: generate(state, resources))
    graph.add_node("faithfulness_check", lambda state: faithfulness_check(state, resources))
    graph.add_node("repair", lambda state: repair_answer(state, resources))
    graph.add_node("abstain", lambda state: abstain(state, resources))

    graph.add_edge(START, "query_analysis")
    graph.add_edge("query_analysis", "retrieve")
    graph.add_edge("retrieve", "context_grade")
    graph.add_conditional_edges(
        "context_grade",
        lambda state: route_after_context_grade(state, resources),
        {"generate": "generate", "rewrite": "query_rewrite", "abstain": "abstain"},
    )
    graph.add_edge("query_rewrite", "retrieve")
    graph.add_edge("generate", "faithfulness_check")
    graph.add_conditional_edges(
        "faithfulness_check",
        lambda state: route_after_faithfulness(state, resources),
        {"end": END, "repair": "repair", "abstain": "abstain"},
    )
    graph.add_edge("repair", "faithfulness_check")
    graph.add_edge("abstain", END)
    return graph.compile()


def initial_state(question: str, ground_truth: str = "", relevant_source_ids: list[str] | None = None) -> AgenticRAGState:
    return {
        "question": question,
        "ground_truth": ground_truth,
        "relevant_source_ids": relevant_source_ids or [],
        "generated_queries": [],
        "retrieved": [],
        "context": "",
        "context_score": 0.0,
        "context_reason": "",
        "answer": "",
        "faithfulness": 0.0,
        "faithfulness_reason": "",
        "retry_count": 0,
        "repair_count": 0,
        "abstained": False,
        "route_trace": [],
        "retrieval_metrics": {},
        "timings_ms": {},
        "llm_usage": usage_to_dict(LLMUsage()),
    }


def query_analysis(state: AgenticRAGState, resources: AgenticRAGResources) -> dict[str, Any]:
    return {
        "generated_queries": [state["question"]],
        "route_trace": append_trace(state, "query_analysis:use_original_query"),
    }


def retrieve(state: AgenticRAGState, resources: AgenticRAGResources) -> dict[str, Any]:
    started = time.perf_counter()
    queries = state.get("generated_queries") or [state["question"]]
    candidates = resources.index.hybrid_search(queries, top_k=resources.first_stage_k)
    ranked = resources.index.rerank(state["question"], candidates, top_k=resources.top_k)
    context = build_context(resources.index.chunks, ranked)
    retrieved = []
    for idx, score in ranked:
        chunk = resources.index.chunks[idx]
        retrieved.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "title": chunk.title,
                "path": chunk.path,
                "score": score,
            }
        )
    source_ids = unique_source_ids(resources.index.chunks, [idx for idx, _ in ranked])
    metrics = retrieval_metrics(source_ids, set(state.get("relevant_source_ids", [])))
    return {
        "retrieved": retrieved,
        "context": context,
        "retrieval_metrics": metrics,
        "timings_ms": add_timing(state, "retrieve_ms", elapsed_ms(started)),
        "route_trace": append_trace(state, f"retrieve:{','.join(source_ids) or 'none'}"),
    }


def context_grade(state: AgenticRAGState, resources: AgenticRAGResources) -> dict[str, Any]:
    started = time.perf_counter()
    if not state.get("context"):
        return {
            "context_score": 0.0,
            "context_reason": "没有检索到上下文。",
            "route_trace": append_trace(state, "context_grade:empty"),
        }

    prompt = f"""请评估参考资料是否足以回答问题。
只输出 JSON：{{"score": 0.0到1.0, "reason": "一句中文理由"}}

评分标准：
- 1.0：资料直接覆盖问题所有关键点
- 0.7：资料覆盖主要问题，少量细节不足
- 0.4：资料只有部分相关
- 0.0：资料基本无关

问题：{state['question']}

参考资料：
{state['context']}
"""
    text, usage = resources.llm.complete(prompt, temperature=0.0)
    score, reason = parse_score_reason(text)
    return {
        "context_score": score,
        "context_reason": reason,
        "llm_usage": add_usage(state, usage),
        "timings_ms": add_timing(state, "context_grade_ms", elapsed_ms(started)),
        "route_trace": append_trace(state, f"context_grade:{score:.2f}"),
    }


def query_rewrite(state: AgenticRAGState, resources: AgenticRAGResources) -> dict[str, Any]:
    started = time.perf_counter()
    queries, usage = transform_query(resources.llm, state["question"])
    retry_count = int(state.get("retry_count", 0)) + 1
    return {
        "generated_queries": queries,
        "retry_count": retry_count,
        "llm_usage": add_usage(state, usage),
        "timings_ms": add_timing(state, "query_rewrite_ms", elapsed_ms(started)),
        "route_trace": append_trace(state, f"query_rewrite:{retry_count}"),
    }


def generate(state: AgenticRAGState, resources: AgenticRAGResources) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = f"""你是严谨的 Agentic RAG 回答节点。请只基于参考资料回答问题。

要求：
1. 如果资料不足，明确说资料不足。
2. 用 source_id 或来源编号标注关键依据。
3. 不要补充参考资料之外的事实。

问题：{state['question']}

参考资料：
{state['context']}

回答："""
    answer, usage = resources.llm.complete(prompt, temperature=0.15)
    return {
        "answer": answer,
        "llm_usage": add_usage(state, usage),
        "timings_ms": add_timing(state, "generate_ms", elapsed_ms(started)),
        "route_trace": append_trace(state, "generate"),
    }


def faithfulness_check(state: AgenticRAGState, resources: AgenticRAGResources) -> dict[str, Any]:
    started = time.perf_counter()
    eval_item = EvalQuestion(
        question=state["question"],
        relevant_source_ids=tuple(state.get("relevant_source_ids", [])),
        ground_truth=state.get("ground_truth") or "没有人工参考答案；只根据上下文判断回答忠实度。",
        evidence="Phase3 Agentic RAG runtime faithfulness check.",
    )
    score, reason, usage = judge_faithfulness(resources.llm, eval_item, state.get("context", ""), state.get("answer", ""))
    return {
        "faithfulness": score,
        "faithfulness_reason": reason,
        "llm_usage": add_usage(state, usage),
        "timings_ms": add_timing(state, "faithfulness_ms", elapsed_ms(started)),
        "route_trace": append_trace(state, f"faithfulness_check:{score:.2f}"),
    }


def repair_answer(state: AgenticRAGState, resources: AgenticRAGResources) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = f"""下面回答的忠实度检查未通过，请修复。

要求：
1. 删除所有参考资料无法支持的声明。
2. 只保留能从参考资料中找到依据的内容。
3. 如果资料不足，直接说明资料不足。

问题：{state['question']}

参考资料：
{state['context']}

原回答：
{state.get('answer', '')}

忠实度问题：
{state.get('faithfulness_reason', '')}

修复后的回答："""
    answer, usage = resources.llm.complete(prompt, temperature=0.0)
    repair_count = int(state.get("repair_count", 0)) + 1
    return {
        "answer": answer,
        "repair_count": repair_count,
        "llm_usage": add_usage(state, usage),
        "timings_ms": add_timing(state, "repair_ms", elapsed_ms(started)),
        "route_trace": append_trace(state, f"repair:{repair_count}"),
    }


def abstain(state: AgenticRAGState, resources: AgenticRAGResources) -> dict[str, Any]:
    return {
        "answer": "根据当前检索到的资料，我无法可靠回答这个问题。建议补充更相关的资料或换一个更具体的问题。",
        "faithfulness": 1.0,
        "faithfulness_reason": "资料不足时选择拒答，没有编造未被上下文支持的信息。",
        "abstained": True,
        "route_trace": append_trace(state, "abstain"),
    }


def route_after_context_grade(state: AgenticRAGState, resources: AgenticRAGResources) -> str:
    if float(state.get("context_score", 0.0)) >= resources.min_context_score:
        return "generate"
    if int(state.get("retry_count", 0)) < resources.max_retries:
        return "rewrite"
    return "abstain"


def route_after_faithfulness(state: AgenticRAGState, resources: AgenticRAGResources) -> str:
    score = float(state.get("faithfulness", 0.0))
    if score >= resources.min_faithfulness:
        return "end"
    if int(state.get("repair_count", 0)) < resources.max_repairs:
        return "repair"
    if score < 0.45:
        return "abstain"
    return "end"


def retrieval_metrics(retrieved_source_ids: list[str], relevant: set[str]) -> dict[str, float]:
    if not relevant:
        return {"precision_at_3": 0.0, "recall_at_3": 0.0, "mrr": 0.0, "ndcg_at_3": 0.0}
    return {
        "precision_at_3": precision_at_k(retrieved_source_ids, relevant, 3),
        "recall_at_3": recall_at_k(retrieved_source_ids, relevant, 3),
        "mrr": mean_reciprocal_rank(retrieved_source_ids, relevant),
        "ndcg_at_3": ndcg_at_k(retrieved_source_ids, relevant, 3),
    }


def parse_score_reason(text: str) -> tuple[float, str]:
    try:
        payload = json.loads(extract_json(text))
        return clamp(float(payload.get("score", 0.0))), str(payload.get("reason", "")).strip()
    except Exception:
        match = re.search(r"([01](?:\.\d+)?)", text)
        if match:
            return clamp(float(match.group(1))), text[:120]
        return 0.0, f"评分解析失败：{text[:120]}"


def extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1]
    return text


def clamp(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def append_trace(state: AgenticRAGState, item: str) -> list[str]:
    return [*state.get("route_trace", []), item]


def add_timing(state: AgenticRAGState, key: str, value: float) -> dict[str, float]:
    timings = dict(state.get("timings_ms", {}))
    timings[key] = timings.get(key, 0.0) + value
    return timings


def add_usage(state: AgenticRAGState, usage: LLMUsage) -> dict[str, float]:
    current = usage_from_dict(state.get("llm_usage", {}))
    current.add(usage)
    return usage_to_dict(current)


def usage_from_dict(payload: dict[str, Any]) -> LLMUsage:
    return LLMUsage(
        prompt_tokens=int(payload.get("prompt_tokens", 0)),
        completion_tokens=int(payload.get("completion_tokens", 0)),
        total_tokens=int(payload.get("total_tokens", 0)),
        cost_usd=float(payload.get("cost_usd", 0.0)),
        calls=int(payload.get("calls", 0)),
    )


def usage_to_dict(usage: LLMUsage) -> dict[str, float]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cost_usd": usage.cost_usd,
        "calls": usage.calls,
    }


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def total_latency_ms(state: AgenticRAGState) -> float:
    return sum(float(value) for value in state.get("timings_ms", {}).values())
