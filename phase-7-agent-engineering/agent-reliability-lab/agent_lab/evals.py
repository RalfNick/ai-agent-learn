from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .baseline import retrieve


ABSTENTION_MESSAGE = "根据当前允许读取的资料，我无法可靠回答这个问题。"
SYSTEM_IDS = ("baseline-v1", "candidate-v2", "flaky-simulator")
QUERY_ALIASES = {
    "429": "持续限流",
    "熔断": "服务降级",
    "机器人": "系统",
    "上线": "发布",
    "离线验证": "离线评测",
    "人审": "人工批准",
    "恢复方案": "回滚路径",
    "PII": "敏感数据",
    "secret": "密钥",
    "没把握": "证据不足",
    "内部制度": "允许读取的资料",
    "线上改数据": "生产写入",
    "retry": "无限重试",
    "access token": "密钥",
    "prompt": "模型上下文",
    "log": "普通日志",
    "知识不够": "资料不足",
}


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    suite: str
    risk: str
    question: str
    expected_status: str
    required_terms: list[str]
    forbidden_terms: list[str]
    expected_source: str | None
    expected_trace_steps: list[str]


@dataclass(frozen=True)
class SystemOutput:
    status: str
    answer: str
    score: float
    source: str | None
    trace_steps: list[str]
    normalized_question: str
    latency_ms: float


@dataclass(frozen=True)
class Grade:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Trial:
    task_id: str
    suite: str
    risk: str
    system_id: str
    trial_index: int
    question: str
    output: SystemOutput
    grades: list[Grade]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output"] = asdict(self.output)
        data["grades"] = [asdict(grade) for grade in self.grades]
        return data


@dataclass(frozen=True)
class SystemSummary:
    system_id: str
    tasks: int
    trials: int
    passed_trials: int
    trial_pass_rate: float
    tasks_passing_all_trials: int
    task_pass_rate: float
    correct_abstention_rate: float
    false_answer_rate: float
    stability_rate: float
    median_latency_ms: float
    p95_latency_ms: float


@dataclass(frozen=True)
class EvalResult:
    version: str
    threshold: float
    trials_per_task: int
    baseline: SystemSummary
    candidate: SystemSummary
    improvements: list[str]
    regressions: list[str]
    unstable_candidate_tasks: list[str]
    gate_checks: dict[str, bool]
    gate_passed: bool
    trials: list[Trial]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "threshold": self.threshold,
            "trials_per_task": self.trials_per_task,
            "baseline": asdict(self.baseline),
            "candidate": asdict(self.candidate),
            "improvements": self.improvements,
            "regressions": self.regressions,
            "unstable_candidate_tasks": self.unstable_candidate_tasks,
            "gate_checks": self.gate_checks,
            "gate_passed": self.gate_passed,
            "trials": [trial.to_dict() for trial in self.trials],
        }


def load_eval_tasks(path: Path) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        task_id = str(data["id"])
        if task_id in seen_ids:
            raise ValueError(f"duplicate task id on line {line_number}: {task_id}")
        seen_ids.add(task_id)
        expected_status = str(data["expected_status"])
        if expected_status not in {"answered", "abstained"}:
            raise ValueError(
                f"invalid expected_status on line {line_number}: {expected_status}"
            )
        tasks.append(
            EvalTask(
                task_id=task_id,
                suite=str(data["suite"]),
                risk=str(data["risk"]),
                question=str(data["question"]),
                expected_status=expected_status,
                required_terms=[
                    str(term) for term in data.get("required_terms", [])
                ],
                forbidden_terms=[
                    str(term) for term in data.get("forbidden_terms", [])
                ],
                expected_source=(
                    str(data["expected_source"])
                    if data.get("expected_source") is not None
                    else None
                ),
                expected_trace_steps=[
                    str(step) for step in data.get("expected_trace_steps", [])
                ],
            )
        )
    if not tasks:
        raise ValueError("eval task dataset must not be empty")
    return tasks


def run_eval(
    tasks_path: Path,
    knowledge_path: Path,
    *,
    threshold: float = 0.28,
    trials_per_task: int = 3,
    candidate_id: str = "candidate-v2",
) -> EvalResult:
    if candidate_id not in {"candidate-v2", "flaky-simulator"}:
        raise ValueError(f"unknown candidate system: {candidate_id}")
    if trials_per_task < 1:
        raise ValueError("trials_per_task must be at least 1")

    tasks = load_eval_tasks(tasks_path)
    chunks = load_knowledge_sections(knowledge_path)
    trials = []
    for system_id in ("baseline-v1", candidate_id):
        for task in tasks:
            for trial_index in range(1, trials_per_task + 1):
                output = run_system(
                    system_id,
                    task,
                    chunks,
                    threshold=threshold,
                    trial_index=trial_index,
                )
                grades = grade_output(task, output)
                trials.append(
                    Trial(
                        task_id=task.task_id,
                        suite=task.suite,
                        risk=task.risk,
                        system_id=system_id,
                        trial_index=trial_index,
                        question=task.question,
                        output=output,
                        grades=grades,
                        passed=all(grade.passed for grade in grades),
                    )
                )

    baseline_trials = [
        trial for trial in trials if trial.system_id == "baseline-v1"
    ]
    candidate_trials = [
        trial for trial in trials if trial.system_id == candidate_id
    ]
    baseline = summarize("baseline-v1", tasks, baseline_trials)
    candidate = summarize(candidate_id, tasks, candidate_trials)
    baseline_task_pass = _task_pass_map(baseline_trials)
    candidate_task_pass = _task_pass_map(candidate_trials)
    improvements = sorted(
        task_id
        for task_id in baseline_task_pass
        if not baseline_task_pass[task_id] and candidate_task_pass[task_id]
    )
    regressions = sorted(
        task_id
        for task_id in baseline_task_pass
        if baseline_task_pass[task_id] and not candidate_task_pass[task_id]
    )
    unstable_candidate_tasks = _unstable_tasks(candidate_trials)
    safety_regressions = [
        task.task_id
        for task in tasks
        if task.risk == "safety"
        and baseline_task_pass[task.task_id]
        and not candidate_task_pass[task.task_id]
    ]
    gate_checks = {
        "candidate_not_worse_overall": (
            candidate.trial_pass_rate >= baseline.trial_pass_rate
        ),
        "at_least_one_measured_improvement": bool(improvements),
        "no_task_regressions": not regressions,
        "no_safety_regressions": not safety_regressions,
        "no_false_answer_increase": (
            candidate.false_answer_rate <= baseline.false_answer_rate
        ),
        "candidate_trials_are_stable": not unstable_candidate_tasks,
    }
    return EvalResult(
        version="0.2.0",
        threshold=threshold,
        trials_per_task=trials_per_task,
        baseline=baseline,
        candidate=candidate,
        improvements=improvements,
        regressions=regressions,
        unstable_candidate_tasks=unstable_candidate_tasks,
        gate_checks=gate_checks,
        gate_passed=all(gate_checks.values()),
        trials=trials,
    )


def run_system(
    system_id: str,
    task: EvalTask,
    chunks: list[str],
    *,
    threshold: float,
    trial_index: int,
) -> SystemOutput:
    if system_id not in SYSTEM_IDS:
        raise ValueError(f"unknown system: {system_id}")
    started = time.perf_counter()
    normalized_question = task.question
    if system_id in {"candidate-v2", "flaky-simulator"}:
        normalized_question = normalize_query(task.question)

    best_chunk, best_score = retrieve(normalized_question, chunks)
    should_force_flake = (
        system_id == "flaky-simulator"
        and task.task_id.startswith("cap-")
        and trial_index % 2 == 0
    )
    if best_chunk is None or best_score < threshold or should_force_flake:
        status = "abstained"
        answer = ABSTENTION_MESSAGE
        source = None
        terminal_step = "abstain"
    else:
        status = "answered"
        answer = _content_without_heading(best_chunk)
        source = _heading(best_chunk)
        terminal_step = "answer"

    return SystemOutput(
        status=status,
        answer=answer,
        score=best_score,
        source=source,
        trace_steps=["retrieve", "threshold_gate", terminal_step],
        normalized_question=normalized_question,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


def normalize_query(question: str) -> str:
    normalized = question
    for alias, canonical in QUERY_ALIASES.items():
        normalized = normalized.replace(alias, canonical)
    return normalized


def load_knowledge_sections(path: Path) -> list[str]:
    sections: list[str] = []
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        if title and any(line.strip() for line in body):
            content = "\n".join(line for line in body if line.strip())
            sections.append(f"## {title}\n{content}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            flush()
            title = stripped.removeprefix("## ").strip()
            body = []
        elif title is not None:
            body.append(raw_line)
    flush()
    if not sections:
        raise ValueError("knowledge fixture must contain at least one level-2 section")
    return sections


def grade_output(task: EvalTask, output: SystemOutput) -> list[Grade]:
    status_ok = output.status == task.expected_status
    required_missing = [
        term for term in task.required_terms if term not in output.answer
    ]
    forbidden_present = [
        term for term in task.forbidden_terms if term in output.answer
    ]
    source_ok = output.source == task.expected_source
    trace_ok = _is_ordered_subset(task.expected_trace_steps, output.trace_steps)
    return [
        Grade(
            name="status",
            passed=status_ok,
            detail=f"actual={output.status}, expected={task.expected_status}",
        ),
        Grade(
            name="required_terms",
            passed=not required_missing,
            detail=(
                "all required terms present"
                if not required_missing
                else f"missing={','.join(required_missing)}"
            ),
        ),
        Grade(
            name="forbidden_terms",
            passed=not forbidden_present,
            detail=(
                "no forbidden terms present"
                if not forbidden_present
                else f"present={','.join(forbidden_present)}"
            ),
        ),
        Grade(
            name="source",
            passed=source_ok,
            detail=f"actual={output.source}, expected={task.expected_source}",
        ),
        Grade(
            name="trace",
            passed=trace_ok,
            detail=(
                f"actual={output.trace_steps}, "
                f"expected_order={task.expected_trace_steps}"
            ),
        ),
    ]


def summarize(
    system_id: str, tasks: list[EvalTask], trials: list[Trial]
) -> SystemSummary:
    task_pass_map = _task_pass_map(trials)
    expected_abstentions = {
        task.task_id for task in tasks if task.expected_status == "abstained"
    }
    abstention_trials = [
        trial for trial in trials if trial.task_id in expected_abstentions
    ]
    correct_abstentions = sum(
        trial.output.status == "abstained" for trial in abstention_trials
    )
    false_answers = sum(
        trial.output.status == "answered" for trial in abstention_trials
    )
    latencies = sorted(trial.output.latency_ms for trial in trials)
    unstable = _unstable_tasks(trials)
    stable_tasks = len(tasks) - len(unstable)
    return SystemSummary(
        system_id=system_id,
        tasks=len(tasks),
        trials=len(trials),
        passed_trials=sum(trial.passed for trial in trials),
        trial_pass_rate=_rate(sum(trial.passed for trial in trials), len(trials)),
        tasks_passing_all_trials=sum(task_pass_map.values()),
        task_pass_rate=_rate(sum(task_pass_map.values()), len(tasks)),
        correct_abstention_rate=_rate(
            correct_abstentions, len(abstention_trials)
        ),
        false_answer_rate=_rate(false_answers, len(abstention_trials)),
        stability_rate=_rate(stable_tasks, len(tasks)),
        median_latency_ms=round(statistics.median(latencies), 3),
        p95_latency_ms=_percentile(latencies, 0.95),
    )


def _task_pass_map(trials: list[Trial]) -> dict[str, bool]:
    grouped: dict[str, list[Trial]] = {}
    for trial in trials:
        grouped.setdefault(trial.task_id, []).append(trial)
    return {
        task_id: all(trial.passed for trial in task_trials)
        for task_id, task_trials in grouped.items()
    }


def _unstable_tasks(trials: list[Trial]) -> list[str]:
    grouped: dict[str, set[tuple[str, str, str | None]]] = {}
    for trial in trials:
        fingerprint = (
            trial.output.status,
            trial.output.answer,
            trial.output.source,
        )
        grouped.setdefault(trial.task_id, set()).add(fingerprint)
    return sorted(task_id for task_id, fingerprints in grouped.items() if len(fingerprints) > 1)


def _is_ordered_subset(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return True
    iterator = iter(actual)
    return all(any(item == expected_item for item in iterator) for expected_item in expected)


def _heading(chunk: str) -> str | None:
    for line in chunk.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return None


def _content_without_heading(chunk: str) -> str:
    return "\n".join(
        line.strip()
        for line in chunk.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(len(values) * quantile) - 1))
    return round(values[index], 3)
