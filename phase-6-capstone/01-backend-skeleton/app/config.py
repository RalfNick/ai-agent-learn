from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    service_name: str = "phase6-capstone-api"
    phase: str = "phase-6"
    version: str = "0.1.0"
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:3020",
        "http://localhost:3020",
    )


def get_settings() -> Settings:
    return Settings()
