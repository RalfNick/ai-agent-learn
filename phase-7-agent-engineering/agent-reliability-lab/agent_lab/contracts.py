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
    if not isinstance(data, dict):
        raise ContractError("contract must be an object")

    missing_sections = sorted(REQUIRED_SECTIONS - data.keys())
    if missing_sections:
        raise ContractError(f"missing sections: {', '.join(missing_sections)}")

    for field in ("id", "version"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise ContractError(f"{field} must be a non-empty string")

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

    trusted_sources = _string_set(data["input"]["trusted_sources"], "input.trusted_sources")
    untrusted_sources = _string_set(
        data["input"]["untrusted_sources"], "input.untrusted_sources"
    )
    _reject_overlap(
        trusted_sources,
        untrusted_sources,
        "input sources cannot be both trusted and untrusted",
    )

    allowed_actions = _string_set(
        data["boundaries"]["allowed_actions"], "boundaries.allowed_actions"
    )
    prohibited_actions = _string_set(
        data["boundaries"]["prohibited_actions"], "boundaries.prohibited_actions"
    )
    approval_required = _string_set(
        data["boundaries"]["approval_required"], "boundaries.approval_required"
    )
    _reject_overlap(
        allowed_actions,
        prohibited_actions,
        "actions cannot be both allowed and prohibited",
    )
    _reject_overlap(
        prohibited_actions,
        approval_required,
        "actions cannot be both prohibited and approval-gated",
    )

    success_states = _string_set(
        data["done"]["terminal_states"], "done.terminal_states"
    )
    failure_states = _string_set(
        data["failure"]["terminal_states"], "failure.terminal_states"
    )
    _reject_overlap(
        success_states,
        failure_states,
        "terminal states cannot be both successful and failed",
    )


def _string_set(value: Any, field: str) -> set[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ContractError(f"{field} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ContractError(f"{field} must not contain duplicates")
    return set(value)


def _reject_overlap(left: set[str], right: set[str], message: str) -> None:
    overlap = sorted(left & right)
    if overlap:
        raise ContractError(f"{message}: {', '.join(overlap)}")


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}
