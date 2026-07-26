from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = {
    "id",
    "version",
    "job",
    "input",
    "done",
    "boundaries",
    "failure",
    "evidence",
    "baseline",
}

REQUIRED_FIELDS = {
    "job": {"actor", "task", "why_agent"},
    "input": {"required", "trusted_sources", "untrusted_sources"},
    "done": {"terminal_states", "validators"},
    "boundaries": {"allowed_actions", "prohibited_actions", "approval_required"},
    "failure": {"terminal_states", "handoff_when"},
    "evidence": {"dataset", "primary_metrics", "secondary_metrics"},
    "baseline": {"strategy", "command", "agent_required_if"},
}


class ContractError(ValueError):
    """Raised when an Agent System Card is incomplete."""


def load_contract(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_contract(data)
    return data


def validate_contract(data: dict[str, Any]) -> None:
    missing_sections = sorted(REQUIRED_SECTIONS - data.keys())
    if missing_sections:
        raise ContractError(f"missing sections: {', '.join(missing_sections)}")

    for section, required_fields in REQUIRED_FIELDS.items():
        value = data.get(section)
        if not isinstance(value, dict):
            raise ContractError(f"{section} must be an object")
        missing_fields = sorted(required_fields - value.keys())
        if missing_fields:
            raise ContractError(
                f"{section} missing fields: {', '.join(missing_fields)}"
            )
        for field in required_fields:
            if _is_empty(value[field]):
                raise ContractError(f"{section}.{field} must not be empty")


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}
