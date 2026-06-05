from __future__ import annotations

import re
import sys
from pathlib import Path

from app.config import Settings
from app.schemas import (
    AnswerResponse,
    MemoryContextItem,
    ReviewResponse,
    ToolResultItem,
)


def _add_phase4_runtime_to_path(project_root: Path) -> None:
    runtime_root = project_root / "phase-4-advanced" / "05-agent-runtime-integration"
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))


class RuntimeAdapter:
    """Thin API adapter around the Phase4 deterministic runtime."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        _add_phase4_runtime_to_path(settings.project_root)

        from runtime import IntegratedAgentRuntime

        self._runtime_cls = IntegratedAgentRuntime

    def answer(self, question: str, session_id: str) -> AnswerResponse:
        memory_path = self._memory_path(session_id)
        runtime = self._runtime_cls(project_root=self.settings.project_root, memory_path=memory_path)
        result = runtime.answer(question)

        return AnswerResponse(
            question=result.question,
            session_id=session_id,
            answer=result.answer,
            memory_context=[self._memory_item(item) for item in result.memory_context],
            written_memory=self._memory_item(result.written_memory) if result.written_memory else None,
            tool_results=[
                ToolResultItem(
                    tool_name=item.tool_name,
                    query=item.query,
                    count=item.count,
                    evidence=item.evidence,
                    summary=item.summary,
                )
                for item in result.tool_results
            ],
            evidence=result.evidence,
            review=ReviewResponse(
                status=result.review.status.value,
                score=result.review.score,
                comments=result.review.comments,
            ),
            trace=result.trace,
        )

    def _memory_path(self, session_id: str) -> Path:
        safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id.strip())[:80] or "default"
        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        return self.settings.memory_dir / f"{safe_session}.json"

    def _memory_item(self, memory) -> MemoryContextItem:
        return MemoryContextItem(
            memory_type=memory.memory_type.value,
            subject=memory.subject,
            content=memory.content,
        )
