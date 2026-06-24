from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from knowledge import LocalKnowledgeIndex
from knowledge.models import RetrievalResult

from .evidence import (
    answer_is_evidence_supported,
    build_evidence_answer,
    context_score,
    results_to_sources,
)
from .models import QASource, QATraceStep


AnswerBuilder = Callable[[str, Sequence[RetrievalResult]], str]


class QAWorkflowState(TypedDict, total=False):
    question: str
    session_id: str
    results: list[RetrievalResult]
    context_score: float
    answer: str
    sources: list[QASource]
    trace: list[QATraceStep]
    review_status: str
    repair_count: int


@dataclass(frozen=True)
class WorkflowResources:
    index: LocalKnowledgeIndex
    min_context_score: float
    top_k: int
    max_repairs: int
    answer_builder: AnswerBuilder


def build_qa_workflow(resources: WorkflowResources):
    graph = StateGraph(QAWorkflowState)
    graph.add_node("retrieve", lambda state: retrieve(state, resources))
    graph.add_node("context_grade", lambda state: grade_context(state, resources))
    graph.add_node("answer_generate", lambda state: generate_answer(state, resources))
    graph.add_node("evidence_review", lambda state: review_evidence(state, resources))
    graph.add_node("repair", lambda state: repair_answer(state, resources))
    graph.add_node("abstain", lambda state: abstain(state, resources))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "context_grade")
    graph.add_conditional_edges(
        "context_grade",
        lambda state: route_after_context_grade(state, resources),
        {"answer": "answer_generate", "abstain": "abstain"},
    )
    graph.add_edge("answer_generate", "evidence_review")
    graph.add_conditional_edges(
        "evidence_review",
        lambda state: route_after_review(state, resources),
        {"end": END, "repair": "repair", "abstain": "abstain"},
    )
    graph.add_edge("repair", "evidence_review")
    graph.add_edge("abstain", END)
    return graph.compile()


def retrieve(state: QAWorkflowState, resources: WorkflowResources) -> dict:
    started = time.perf_counter()
    results = resources.index.search(state["question"], limit=resources.top_k)
    return {
        "results": results,
        "trace": append_trace(
            state,
            QATraceStep(
                step="retrieve",
                detail=f"Retrieved {len(results)} candidate chunks.",
                latency_ms=elapsed_ms(started),
            ),
        ),
    }


def grade_context(state: QAWorkflowState, resources: WorkflowResources) -> dict:
    score = context_score(state.get("results", []))
    return {
        "context_score": score,
        "trace": append_trace(
            state,
            QATraceStep(
                step="context_grade",
                detail=f"Context score {score:.3f}; threshold {resources.min_context_score:.3f}.",
            ),
        ),
    }


def generate_answer(state: QAWorkflowState, resources: WorkflowResources) -> dict:
    results = state.get("results", [])
    return {
        "answer": resources.answer_builder(state["question"], results),
        "sources": results_to_sources(results),
        "trace": append_trace(
            state,
            QATraceStep(step="answer.generate", detail="Generated answer from retrieved evidence."),
        ),
    }


def review_evidence(state: QAWorkflowState, resources: WorkflowResources) -> dict:
    supported = answer_is_evidence_supported(state.get("answer", ""), state.get("sources", []))
    step = "review.evidence_supported" if supported else "review.failed"
    detail = (
        "Answer only uses snippets from retrieved sources."
        if supported
        else "Answer contains lines without a supported source."
    )
    return {
        "review_status": "evidence_supported" if supported else "review_failed",
        "trace": append_trace(state, QATraceStep(step=step, detail=detail)),
    }


def repair_answer(state: QAWorkflowState, resources: WorkflowResources) -> dict:
    repair_count = int(state.get("repair_count", 0)) + 1
    repaired_answer = build_evidence_answer(state["question"], state.get("results", []))
    return {
        "answer": repaired_answer,
        "sources": results_to_sources(state.get("results", [])),
        "repair_count": repair_count,
        "trace": append_trace(
            state,
            QATraceStep(
                step="answer.repair",
                detail=f"Rebuilt evidence-only answer; repair_count={repair_count}.",
            ),
        ),
    }


def abstain(state: QAWorkflowState, resources: WorkflowResources) -> dict:
    return {
        "answer": "根据当前知识库资料，我无法可靠回答这个问题。请补充更相关的资料后再试。",
        "sources": [],
        "review_status": "abstained",
        "trace": append_trace(
            state,
            QATraceStep(step="abstain", detail="Workflow refused to answer with weak evidence."),
        ),
    }


def route_after_context_grade(state: QAWorkflowState, resources: WorkflowResources) -> str:
    if float(state.get("context_score", 0.0)) < resources.min_context_score:
        return "abstain"
    return "answer"


def route_after_review(state: QAWorkflowState, resources: WorkflowResources) -> str:
    if state.get("review_status") == "evidence_supported":
        return "end"
    if int(state.get("repair_count", 0)) < resources.max_repairs:
        return "repair"
    return "abstain"


def append_trace(state: QAWorkflowState, step: QATraceStep) -> list[QATraceStep]:
    return [*state.get("trace", []), step]


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
