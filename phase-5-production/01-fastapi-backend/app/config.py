from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    service_name: str = "phase5-agent-api"
    phase: str = "phase-5"
    version: str = "0.1.0"
    project_root: Path = Path(__file__).resolve().parents[3]
    memory_dir: Path = Path(__file__).resolve().parents[1] / ".memory"


def get_settings() -> Settings:
    return Settings()
