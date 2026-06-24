from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
CAPSTONE_ROOT = RUNTIME_ROOT.parents[0]
INGESTION_ROOT = CAPSTONE_ROOT / "02-knowledge-ingestion"
if str(INGESTION_ROOT) not in sys.path:
    sys.path.insert(0, str(INGESTION_ROOT))

from knowledge import LocalKnowledgeIndex, build_index_from_paths

from .evidence import build_evidence_answer
from .models import QAResponse, QATraceStep
from .workflow import AnswerBuilder, WorkflowResources, build_qa_workflow


class AgenticQARuntime:
    def __init__(
        self,
        index: LocalKnowledgeIndex,
        min_context_score: float = 0.25,
        top_k: int = 3,
        unsafe_answer_builder: AnswerBuilder | None = None,
        max_repairs: int = 1,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_repairs < 0:
            raise ValueError("max_repairs must not be negative")
        self.index = index
        self.min_context_score = min_context_score
        self.top_k = top_k
        self.max_repairs = max_repairs
        self.answer_builder = unsafe_answer_builder or build_evidence_answer
        self._app = build_qa_workflow(
            WorkflowResources(
                index=index,
                min_context_score=min_context_score,
                top_k=top_k,
                max_repairs=max_repairs,
                answer_builder=self.answer_builder,
            )
        )

    def answer(self, question: str, session_id: str = "default") -> QAResponse:
        final_state = self._app.invoke(
            {
                "question": question,
                "session_id": session_id,
                "repair_count": 0,
                "trace": [QATraceStep(step="request.received", detail="Accepted answer request.")],
            }
        )
        return QAResponse(
            question=question,
            session_id=session_id,
            answer=final_state.get("answer", ""),
            mode="agentic_rag",
            sources=final_state.get("sources", []),
            trace=final_state.get("trace", []),
            review_status=final_state.get("review_status"),
            context_score=float(final_state.get("context_score", 0.0)),
        )


def build_runtime_from_sources(
    paths: Sequence[Path | str],
    min_context_score: float = 0.25,
    top_k: int = 3,
) -> AgenticQARuntime:
    index = build_index_from_paths(paths)
    return AgenticQARuntime(index=index, min_context_score=min_context_score, top_k=top_k)
