"""
Benchmark linear Phase 2 RAG against the LangGraph Agentic RAG workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from agentic_rag_graph import build_agentic_rag_app, build_resources, initial_state, total_latency_ms


ROOT = Path(__file__).resolve().parents[2]
PHASE2_BENCHMARK = ROOT / "phase-2-rag" / "05-rag-benchmark"
sys.path.insert(0, str(PHASE2_BENCHMARK))

from benchmark import LLMUsage, RAGConfig as LinearRAGConfig, run_one as run_linear_one  # noqa: E402
from benchmark_dataset import EVAL_QUESTIONS, EvalQuestion, validate_dataset  # noqa: E402


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
REPORT_DIR = HERE / "reports"
console = Console()


@dataclass
class RunRecord:
    mode: str
    question: str
    precision_at_3: float
    recall_at_3: float
    mrr: float
    ndcg_at_3: float
    faithfulness: float
    latency_ms: float
    cost_usd: float
    llm_calls: int
    retry_count: int = 0
    repair_count: int = 0
    abstained: bool = False
    route_trace: list[str] | None = None
    answer: str = ""


@dataclass
class Summary:
    mode: str
    precision_at_3: float
    recall_at_3: float
    mrr: float
    ndcg_at_3: float
    faithfulness: float
    avg_latency_ms: float
    total_cost_usd: float
    llm_calls: int
    avg_retries: float
    avg_repairs: float
    abstain_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Agentic RAG against Phase2 linear RAG")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N questions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_dataset()
    questions = list(EVAL_QUESTIONS[: args.limit]) if args.limit else list(EVAL_QUESTIONS)
    resources = build_resources()
    app = build_agentic_rag_app(resources)
    linear_config = LinearRAGConfig(
        "linear_hybrid_rerank",
        use_hybrid=True,
        use_rerank=True,
        use_query_transform=False,
    )

    console.print(Panel(f"Phase3 Agentic RAG benchmark | questions={len(questions)}", style="bold blue"))
    records: list[RunRecord] = []
    total = len(questions) * 2
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Running benchmark", total=total)
        for question in questions:
            progress.update(task, description=f"linear: {question.question[:36]}")
            linear = run_linear_one(resources.index, resources.llm, linear_config, question)
            records.append(
                RunRecord(
                    mode="linear_hybrid_rerank",
                    question=question.question,
                    precision_at_3=linear.precision_at_3,
                    recall_at_3=linear.recall_at_3,
                    mrr=linear.mrr,
                    ndcg_at_3=linear.ndcg_at_3,
                    faithfulness=linear.faithfulness,
                    latency_ms=linear.latency_ms,
                    cost_usd=linear.llm_usage.cost_usd,
                    llm_calls=linear.llm_usage.calls,
                    answer=linear.answer,
                )
            )
            progress.advance(task)

            progress.update(task, description=f"agentic: {question.question[:36]}")
            agentic_state = app.invoke(
                initial_state(
                    question.question,
                    ground_truth=question.ground_truth,
                    relevant_source_ids=list(question.relevant_source_ids),
                )
            )
            metrics = agentic_state.get("retrieval_metrics", {})
            usage = agentic_state.get("llm_usage", {})
            records.append(
                RunRecord(
                    mode="agentic_rag_langgraph",
                    question=question.question,
                    precision_at_3=float(metrics.get("precision_at_3", 0.0)),
                    recall_at_3=float(metrics.get("recall_at_3", 0.0)),
                    mrr=float(metrics.get("mrr", 0.0)),
                    ndcg_at_3=float(metrics.get("ndcg_at_3", 0.0)),
                    faithfulness=float(agentic_state.get("faithfulness", 0.0)),
                    latency_ms=total_latency_ms(agentic_state),
                    cost_usd=float(usage.get("cost_usd", 0.0)),
                    llm_calls=int(usage.get("calls", 0)),
                    retry_count=int(agentic_state.get("retry_count", 0)),
                    repair_count=int(agentic_state.get("repair_count", 0)),
                    abstained=bool(agentic_state.get("abstained", False)),
                    route_trace=list(agentic_state.get("route_trace", [])),
                    answer=str(agentic_state.get("answer", "")),
                )
            )
            progress.advance(task)

    summaries = summarize(records)
    write_outputs(records, summaries)
    print_summary(summaries)
    console.print(f"[green]Wrote[/green] {OUTPUT_DIR / 'agentic_rag_results.json'}")
    console.print(f"[green]Wrote[/green] {OUTPUT_DIR / 'agentic_rag_summary.csv'}")
    console.print(f"[green]Wrote[/green] {REPORT_DIR / 'agentic_rag_experiment_report.md'}")


def summarize(records: list[RunRecord]) -> list[Summary]:
    summaries = []
    for mode in ["linear_hybrid_rerank", "agentic_rag_langgraph"]:
        items = [record for record in records if record.mode == mode]
        summaries.append(
            Summary(
                mode=mode,
                precision_at_3=mean([item.precision_at_3 for item in items]),
                recall_at_3=mean([item.recall_at_3 for item in items]),
                mrr=mean([item.mrr for item in items]),
                ndcg_at_3=mean([item.ndcg_at_3 for item in items]),
                faithfulness=mean([item.faithfulness for item in items]),
                avg_latency_ms=mean([item.latency_ms for item in items]),
                total_cost_usd=sum(item.cost_usd for item in items),
                llm_calls=sum(item.llm_calls for item in items),
                avg_retries=mean([float(item.retry_count) for item in items]),
                avg_repairs=mean([float(item.repair_count) for item in items]),
                abstain_count=sum(1 for item in items if item.abstained),
            )
        )
    return summaries


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def write_outputs(records: list[RunRecord], summaries: list[Summary]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "agentic_rag_results.json").write_text(
        json.dumps(
            {
                "summaries": [asdict(summary) for summary in summaries],
                "records": [asdict(record) for record in records],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "agentic_rag_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))
    (REPORT_DIR / "agentic_rag_experiment_report.md").write_text(
        build_report(records, summaries),
        encoding="utf-8",
    )


def build_report(records: list[RunRecord], summaries: list[Summary]) -> str:
    by_mode = {summary.mode: summary for summary in summaries}
    linear = by_mode["linear_hybrid_rerank"]
    agentic = by_mode["agentic_rag_langgraph"]
    interesting = [
        record for record in records
        if record.mode == "agentic_rag_langgraph"
        and (record.retry_count > 0 or record.repair_count > 0 or record.abstained)
    ][:3]
    trace_blocks = "\n\n".join(
        f"### Trace {index}: {record.question}\n\n"
        f"- Retry: {record.retry_count}\n"
        f"- Repair: {record.repair_count}\n"
        f"- Abstained: {record.abstained}\n"
        f"- Faithfulness: {record.faithfulness:.3f}\n"
        f"- Route: `{' -> '.join(record.route_trace or [])}`"
        for index, record in enumerate(interesting, start=1)
    )
    if not trace_blocks:
        trace_blocks = "本次运行没有触发 rewrite、repair 或 abstain。需要降低阈值或补充更难的问题集。"

    return f"""# Agentic RAG 实验报告：LangGraph 是否真的带来编排价值

## 1. 实验设置

本实验复用 Phase2 benchmark 的真实资料和 30 个标注问题，对比两种系统：

| 系统 | 说明 |
|------|------|
| `linear_hybrid_rerank` | Phase2 最优线性 RAG：Hybrid + Cross-Encoder Rerank |
| `agentic_rag_langgraph` | LangGraph 自适应 RAG：检索评分、条件改写、忠实度检查、回答修复、拒答 |

## 2. 指标结果

| 系统 | P@3 | R@3 | MRR | NDCG@3 | Faithfulness | 平均延迟(ms) | 成本 | LLM调用 | 平均重试 | 平均修复 | 拒答 |
|------|-----|-----|-----|--------|--------------|--------------|------|---------|----------|----------|------|
| `{linear.mode}` | {linear.precision_at_3:.3f} | {linear.recall_at_3:.3f} | {linear.mrr:.3f} | {linear.ndcg_at_3:.3f} | {linear.faithfulness:.3f} | {linear.avg_latency_ms:.0f} | ${linear.total_cost_usd:.4f} | {linear.llm_calls} | {linear.avg_retries:.2f} | {linear.avg_repairs:.2f} | {linear.abstain_count} |
| `{agentic.mode}` | {agentic.precision_at_3:.3f} | {agentic.recall_at_3:.3f} | {agentic.mrr:.3f} | {agentic.ndcg_at_3:.3f} | {agentic.faithfulness:.3f} | {agentic.avg_latency_ms:.0f} | ${agentic.total_cost_usd:.4f} | {agentic.llm_calls} | {agentic.avg_retries:.2f} | {agentic.avg_repairs:.2f} | {agentic.abstain_count} |

## 3. 关键观察

- 检索指标主要由 Phase2 的 `hybrid_rerank` 决定，Agentic RAG 不应为了“更 Agent”牺牲默认检索质量。
- Agentic RAG 的价值在生成后质量控制：通过 Faithfulness check、repair 和 abstain 把低可信回答显式暴露出来。
- 代价是延迟和 LLM 调用数上升。是否采用 Agentic RAG，应由问题风险、答案可靠性要求和成本预算决定。

## 4. 自适应 Trace 样例

{trace_blocks}

## 5. 阶段结论

Phase3 的目标不是证明 LangGraph 永远更快，而是掌握 Agent 工作流设计：什么时候检索，什么时候重写，什么时候修复，什么时候拒答。这个 benchmark 把这些决策从 prompt 里的隐式愿望，变成了图里的显式控制流。
"""


def print_summary(summaries: list[Summary]) -> None:
    table = Table(title="Phase3 Agentic RAG Benchmark")
    table.add_column("Mode")
    table.add_column("P@3", justify="right")
    table.add_column("R@3", justify="right")
    table.add_column("NDCG", justify="right")
    table.add_column("Faith", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Calls", justify="right")
    table.add_column("Retry", justify="right")
    table.add_column("Repair", justify="right")
    table.add_column("Abstain", justify="right")
    for summary in summaries:
        table.add_row(
            summary.mode,
            f"{summary.precision_at_3:.3f}",
            f"{summary.recall_at_3:.3f}",
            f"{summary.ndcg_at_3:.3f}",
            f"{summary.faithfulness:.3f}",
            f"{summary.avg_latency_ms:.0f}",
            f"${summary.total_cost_usd:.4f}",
            str(summary.llm_calls),
            f"{summary.avg_retries:.2f}",
            f"{summary.avg_repairs:.2f}",
            str(summary.abstain_count),
        )
    console.print(table)


if __name__ == "__main__":
    main()

