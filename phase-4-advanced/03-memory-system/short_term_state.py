from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShortTermState:
    """Execution state for the current run, similar to what a checkpoint stores."""

    goal: str
    steps: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    pending_actions: list[str] = field(default_factory=list)

    def add_step(self, step: str) -> None:
        self.steps.append(step)

    def add_observation(self, observation: str) -> None:
        self.observations.append(observation)

    def add_pending_action(self, action: str) -> None:
        self.pending_actions.append(action)

    def snapshot(self) -> dict:
        return {
            "goal": self.goal,
            "steps": list(self.steps),
            "observations": list(self.observations),
            "pending_actions": list(self.pending_actions),
        }
