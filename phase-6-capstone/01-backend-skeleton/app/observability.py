from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class ObservabilityStore:
    max_recent_questions: int = 10
    total_answer_requests: int = 0
    last_session_id: str | None = None
    recent_questions: Deque[str] = field(default_factory=lambda: deque(maxlen=10))

    def record_answer_request(self, question: str, session_id: str) -> None:
        self.total_answer_requests += 1
        self.last_session_id = session_id
        self.recent_questions.appendleft(question)

    def summary(self) -> dict:
        return {
            "total_answer_requests": self.total_answer_requests,
            "last_session_id": self.last_session_id,
            "recent_questions": list(self.recent_questions),
        }
