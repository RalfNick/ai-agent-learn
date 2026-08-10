from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


SAFE_ENV_KEYS = {
    "COMSPEC",
    "HOME",
    "LANG",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
}


@dataclass(frozen=True)
class ProcessResult:
    argv: list[str]
    cwd: str
    status: str
    exit_code: int | None
    duration_ms: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def safe_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key.upper() in SAFE_ENV_KEYS}
    if extra:
        env.update(extra)
    return env


def _clip(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit] + "\n...[truncated]", True


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)


def run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float = 30.0,
    output_limit: int = 64_000,
    env: Mapping[str, str] | None = None,
) -> ProcessResult:
    resolved_cwd = cwd.resolve()
    if not resolved_cwd.is_dir():
        raise ValueError(f"working directory does not exist: {resolved_cwd}")
    if not argv:
        raise ValueError("argv cannot be empty")

    started = time.monotonic()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        list(argv),
        cwd=resolved_cwd,
        env=safe_environment(env),
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    status = "completed"
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        status = "timed_out"
        _stop_process(process)
        stdout, stderr = process.communicate()

    clipped_stdout, stdout_truncated = _clip(stdout, output_limit)
    clipped_stderr, stderr_truncated = _clip(stderr, output_limit)
    return ProcessResult(
        argv=list(argv),
        cwd=str(resolved_cwd),
        status=status,
        exit_code=process.returncode,
        duration_ms=round((time.monotonic() - started) * 1000),
        stdout=clipped_stdout,
        stderr=clipped_stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )

