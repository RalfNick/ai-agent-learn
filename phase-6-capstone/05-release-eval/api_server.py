from __future__ import annotations

import sys
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent
CAPSTONE_ROOT = EVAL_ROOT.parents[0]
PROJECT_ROOT = CAPSTONE_ROOT.parents[0]
for path in [
    CAPSTONE_ROOT / "01-backend-skeleton",
    CAPSTONE_ROOT / "02-knowledge-ingestion",
    CAPSTONE_ROOT / "03-agentic-qa-runtime",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentic_qa import build_runtime_from_sources
from app.main import create_app
from app.schemas import AnswerResponse, SourceItem, TraceStep


class AgenticRuntimeAdapter:
    def __init__(self, source_paths) -> None:
        self.runtime = build_runtime_from_sources(source_paths, min_context_score=0.2, top_k=3)

    def answer(self, question: str, session_id: str) -> AnswerResponse:
        response = self.runtime.answer(question=question, session_id=session_id)
        return AnswerResponse(
            question=response.question,
            session_id=response.session_id,
            answer=response.answer,
            mode=response.mode,
            sources=[
                SourceItem(
                    source_id=source.source_id,
                    title=source.title,
                    path=source.path,
                    score=source.score,
                    snippet=source.snippet,
                )
                for source in response.sources
            ],
            trace=[
                TraceStep(step=step.step, detail=step.detail, latency_ms=step.latency_ms)
                for step in response.trace
            ],
            review_status=response.review_status,
        )


app = create_app(runtime=AgenticRuntimeAdapter([PROJECT_ROOT / "docs" / "phase-6"]))
