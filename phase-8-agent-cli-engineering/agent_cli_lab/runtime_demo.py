from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .runner import run_process


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, data: dict[str, object]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def runtime_demo(output: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="agent-cli-runtime-") as temp:
        result = run_process(
            [sys.executable, "-c", "import json; print(json.dumps({'status':'ok','count':3}))"],
            cwd=Path(temp),
            timeout_seconds=5,
        )
    _write_json(output, result.to_dict())
    print(output.resolve())
    return 0 if result.exit_code == 0 else 1


def codex_probe(output: Path, mode: str) -> int:
    codex = shutil.which("codex")
    if not codex:
        print("codex executable was not found on PATH", file=sys.stderr)
        return 3

    with tempfile.TemporaryDirectory(prefix="agent-cli-codex-") as temp:
        fixture = Path(temp) / "workspace"
        artifact_dir = Path(temp) / "artifacts"
        fixture.mkdir()
        artifact_dir.mkdir()
        (fixture / "README.md").write_text("# Disposable Codex Probe\n\nFiles: README.md and data.txt.\n", encoding="utf-8")
        (fixture / "data.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=fixture, check=True)
        subprocess.run(["git", "add", "README.md", "data.txt"], cwd=fixture, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Agent CLI Lab",
                "-c",
                "user.email=agent-cli-lab@example.invalid",
                "commit",
                "-q",
                "-m",
                "add disposable fixture",
            ],
            cwd=fixture,
            check=True,
        )
        final_path = artifact_dir / "final.json"
        schema = ROOT / "schemas" / "codex-probe.schema.json"
        argv = [
            codex,
            "exec",
            "-C",
            str(fixture),
            "--sandbox",
            mode,
            "--ephemeral",
            "--ignore-user-config",
            "--strict-config",
            "--output-schema",
            str(schema),
            "--json",
            "--output-last-message",
            str(final_path),
            "Inspect this disposable repository without changing files. Return JSON with status='ok', file_count, and concise notes.",
        ]
        result = run_process(argv, cwd=fixture, timeout_seconds=180, output_limit=256_000)
        payload = result.to_dict()
        payload["codex_version"] = subprocess.run(
            [codex, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        payload["final_message"] = final_path.read_text(encoding="utf-8") if final_path.exists() else None
        payload["workspace_files_after"] = sorted(path.name for path in fixture.iterdir() if path.name != ".git")
    _write_json(output, payload)
    print(output.resolve())
    return 0 if result.exit_code == 0 and result.status == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent CLI process-contract lab.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("runtime-demo")
    demo.add_argument("--output", type=Path, default=Path("reports/runtime-demo.json"))
    probe = commands.add_parser("codex-probe")
    probe.add_argument("--mode", choices=("read-only",), default="read-only")
    probe.add_argument("--output", type=Path, default=Path("reports/codex-probe.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "runtime-demo":
        return runtime_demo(args.output)
    return codex_probe(args.output, args.mode)
