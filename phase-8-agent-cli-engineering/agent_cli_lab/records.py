from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "fixtures" / "runs.json"


class RunNotFoundError(LookupError):
    pass


def _load(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("run dataset must be a JSON array")
    return data


def list_runs(limit: int = 10, path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    runs = _load(path)
    return {"items": runs[:limit], "count": min(limit, len(runs)), "total": len(runs)}


def get_run(run_id: str, path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    for run in _load(path):
        if run.get("id") == run_id:
            return run
    raise RunNotFoundError(f"unknown run id: {run_id}")


def export_report(run_id: str, output: Path, path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    run = get_run(run_id, path)
    resolved = output.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"run_id": run_id, "output": str(resolved), "bytes": resolved.stat().st_size}

