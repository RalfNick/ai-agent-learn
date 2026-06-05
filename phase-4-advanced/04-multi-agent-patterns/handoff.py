from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents import AgentRole


@dataclass
class HandoffPacket:
    """Explicit contract passed from the supervisor to a specialist agent."""

    target: AgentRole
    task: str
    context: dict[str, Any] = field(default_factory=dict)
    required_outputs: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.value,
            "task": self.task,
            "context": dict(self.context),
            "required_outputs": list(self.required_outputs),
            "constraints": list(self.constraints),
        }


@dataclass
class SupervisorPlan:
    question: str
    handoffs: list[HandoffPacket]
