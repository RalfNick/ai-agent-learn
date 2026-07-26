from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path


CJK_SEQUENCE = re.compile(r"[\u3400-\u9fff]+")
ASCII_WORD = re.compile(r"[a-zA-Z0-9_]{2,}")


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    expected_status: str
    expected_terms: list[str]


@dataclass(frozen=True)
class CaseResult:
    task_id: str
    question: str
    status: str
    answer: str
    score: float
    passed: bool
    failures: list[str]
    latency_ms: float


@dataclass(frozen=True)
class BaselineResult:
    version: str
    strategy: str
    total: int
    passed: int
    task_pass_rate: float
    correct_abstention_rate: float
    cases: list[CaseResult]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "strategy": self.strategy,
            "total": self.total,
            "passed": self.passed,
            "task_pass_rate": self.task_pass_rate,
            "correct_abstention_rate": self.correct_abstention_rate,
            "cases": [asdict(case) for case in self.cases],
        }


def load_tasks(path: Path) -> list[Task]:
    tasks = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        tasks.append(
            Task(
                task_id=str(data["id"]),
                question=str(data["question"]),
                expected_status=str(data["expected_status"]),
                expected_terms=[str(term) for term in data.get("expected_terms", [])],
            )
        )
    return tasks


def load_chunks(path: Path) -> list[str]:
    return [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", path.read_text(encoding="utf-8"))
        if chunk.strip() and not chunk.lstrip().startswith("# ")
    ]


def run_baseline(
    tasks_path: Path,
    knowledge_path: Path,
    threshold: float = 0.28,
) -> BaselineResult:
    tasks = load_tasks(tasks_path)
    chunks = load_chunks(knowledge_path)
    cases = [_run_case(task, chunks, threshold) for task in tasks]
    passed = sum(case.passed for case in cases)
    abstention_cases = [
        case
        for task, case in zip(tasks, cases, strict=True)
        if task.expected_status == "abstained"
    ]
    correct_abstentions = sum(case.status == "abstained" for case in abstention_cases)
    return BaselineResult(
        version="0.1.0",
        strategy="deterministic_paragraph_retrieval",
        total=len(cases),
        passed=passed,
        task_pass_rate=round(passed / len(cases), 4) if cases else 0.0,
        correct_abstention_rate=(
            round(correct_abstentions / len(abstention_cases), 4)
            if abstention_cases
            else 0.0
        ),
        cases=cases,
    )


def _run_case(task: Task, chunks: list[str], threshold: float) -> CaseResult:
    started = time.perf_counter()
    best_chunk, best_score = retrieve(task.question, chunks)
    if best_chunk is None or best_score < threshold:
        status = "abstained"
        answer = "根据当前允许读取的资料，我无法可靠回答这个问题。"
    else:
        status = "answered"
        answer = _content_without_heading(best_chunk)

    failures = []
    if status != task.expected_status:
        failures.append(f"status:{status}!={task.expected_status}")
    for term in task.expected_terms:
        if term not in answer:
            failures.append(f"missing_term:{term}")

    return CaseResult(
        task_id=task.task_id,
        question=task.question,
        status=status,
        answer=answer,
        score=best_score,
        passed=not failures,
        failures=failures,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


def retrieve(question: str, chunks: list[str]) -> tuple[str | None, float]:
    query_tokens = tokenize(question)
    if not query_tokens:
        return None, 0.0

    ranked = []
    for chunk in chunks:
        chunk_tokens = tokenize(chunk)
        overlap = query_tokens & chunk_tokens
        score = len(overlap) / len(query_tokens)
        ranked.append((round(score, 4), chunk))
    if not ranked:
        return None, 0.0
    score, chunk = max(ranked, key=lambda item: item[0])
    return chunk, score


def tokenize(text: str) -> set[str]:
    tokens = {match.group(0).lower() for match in ASCII_WORD.finditer(text)}
    for sequence in CJK_SEQUENCE.findall(text):
        if len(sequence) == 1:
            tokens.add(sequence)
        else:
            tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _content_without_heading(chunk: str) -> str:
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return "\n".join(lines)
