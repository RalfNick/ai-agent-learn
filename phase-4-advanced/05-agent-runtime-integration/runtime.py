from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from project_tools import BenchmarkResult, ProjectToolset, SearchResult


CURRENT_DIR = Path(__file__).resolve().parent
PHASE4_ROOT = CURRENT_DIR.parent
MEMORY_ROOT = PHASE4_ROOT / "03-memory-system"
MULTI_AGENT_ROOT = PHASE4_ROOT / "04-multi-agent-patterns"

for import_root in [str(MEMORY_ROOT), str(MULTI_AGENT_ROOT)]:
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from agents import AgentRole, ReviewResult, SpecialistReport
from long_term_memory import JsonMemoryStore, MemoryRecord
from memory_policy import MemoryPolicy
from short_term_state import ShortTermState
from supervisor import MultiAgentSupervisor, ReviewerAgent


@dataclass
class ToolExecution:
    tool_name: str
    query: str
    count: int
    evidence: list[str]
    summary: str


@dataclass
class RuntimeAnswer:
    question: str
    answer: str
    memory_context: list[MemoryRecord]
    written_memory: MemoryRecord | None
    tool_results: list[ToolExecution]
    evidence: list[str]
    review: ReviewResult
    trace: list[str] = field(default_factory=list)


class IntegratedAgentRuntime:
    """A deterministic Phase4 runtime that joins memory, tools, and multi-agent review."""

    def __init__(self, project_root: Path | str, memory_path: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self.memory_store = JsonMemoryStore(memory_path)
        self.memory_policy = MemoryPolicy()
        self.tools = ProjectToolset(project_root=self.project_root)
        self.supervisor = MultiAgentSupervisor(project_root=self.project_root)
        self.reviewer = ReviewerAgent()

    def answer(self, question: str) -> RuntimeAnswer:
        trace = ["runtime.start"]
        state = ShortTermState(goal=question)
        state.add_step("初始化 Phase4 集成 runtime")

        written_memory = self.memory_policy.extract(question)
        if written_memory is not None:
            self.memory_store.upsert(written_memory)
            trace.append("memory.upsert")
            state.add_step("根据 MemoryPolicy 写入长期记忆")

        memory_context = self.memory_store.search(question, limit=3)
        trace.append("memory.search")
        state.add_step("召回和当前问题相关的长期记忆")

        plan = self.supervisor.plan(question)
        trace.append("supervisor.plan")
        state.add_step("Supervisor 生成 handoff 计划")

        reports: list[SpecialistReport] = []
        tool_results: list[ToolExecution] = []

        for packet in plan.handoffs:
            trace.append(f"handoff.{packet.target.value}")
            reports.append(self.supervisor.specialists[packet.target].handle(packet))
            execution = self._run_tool_for_handoff(packet.target, question)
            if execution is not None:
                tool_results.append(execution)
                trace.append(f"tool.{execution.tool_name}")
            trace.append(f"specialist.{packet.target.value}.report")

        evidence = self._collect_evidence(tool_results, reports)
        answer = self._compose_answer(question, memory_context, tool_results, reports)
        review = self.reviewer.review(answer, evidence)
        trace.append("reviewer.review")

        return RuntimeAnswer(
            question=question,
            answer=answer,
            memory_context=memory_context,
            written_memory=written_memory,
            tool_results=tool_results,
            evidence=evidence,
            review=review,
            trace=trace,
        )

    def _run_tool_for_handoff(self, target: AgentRole, question: str) -> ToolExecution | None:
        if target == AgentRole.DOC_RESEARCHER:
            query = self._doc_query(question)
            result = self.tools.search_docs(query=query, phase="phase-4", limit=5)
            return self._from_search_result("search_docs", query, result)

        if target == AgentRole.CODE_ANALYST:
            query = self._code_query(question)
            result = self.tools.find_code_examples(query=query, phase="phase-4", limit=5)
            return self._from_search_result("find_code_examples", query, result)

        if target == AgentRole.BENCHMARK_AGENT:
            result = self.tools.read_benchmark_summary("phase-3")
            return self._from_benchmark_result(result)

        return None

    def _from_search_result(self, tool_name: str, query: str, result: SearchResult) -> ToolExecution:
        evidence = [hit.path for hit in result.results]
        summary = f"{tool_name}({query}) returned {result.count} hits"
        return ToolExecution(tool_name=tool_name, query=query, count=result.count, evidence=evidence, summary=summary)

    def _from_benchmark_result(self, result: BenchmarkResult) -> ToolExecution:
        evidence = [summary.source for summary in result.summaries]
        row_count = sum(len(summary.rows) for summary in result.summaries)
        summary = f"read_benchmark_summary(phase-3) returned {row_count} rows"
        return ToolExecution(
            tool_name="read_benchmark_summary",
            query="phase-3",
            count=row_count,
            evidence=evidence,
            summary=summary,
        )

    def _collect_evidence(self, tool_results: list[ToolExecution], reports: list[SpecialistReport]) -> list[str]:
        evidence = {item for result in tool_results for item in result.evidence}
        evidence.update(item for report in reports for item in report.evidence)
        return sorted(evidence)

    def _compose_answer(
        self,
        question: str,
        memory_context: list[MemoryRecord],
        tool_results: list[ToolExecution],
        reports: list[SpecialistReport],
    ) -> str:
        lines = [
            f"Phase4 集成回答：{question}",
            "",
            "结论：MCP、Memory 和 Multi-Agent 已经可以组成一个最小可复盘 runtime，进入 Phase5 前可以把它作为生产化原型。",
        ]

        if memory_context:
            lines.append("")
            lines.append("长期记忆上下文：")
            for memory in memory_context:
                lines.append(f"- [{memory.memory_type.value}] {memory.content}")

        lines.append("")
        lines.append("工具证据：")
        for result in tool_results:
            preview = "，".join(result.evidence[:3]) if result.evidence else "无命中"
            lines.append(f"- {result.tool_name}: {result.summary}; evidence={preview}")

        lines.append("")
        lines.append("Specialist 观察：")
        for report in reports:
            lines.append(f"- {report.role.value}: {report.summary}")

        lines.append("")
        lines.append("风险：当前 runtime 仍是确定性学习实现，没有真实 LLM 工具循环、权限审批、服务化部署和线上观测。")
        return "\n".join(lines)

    def _doc_query(self, question: str) -> str:
        if "Memory" in question or "记忆" in question:
            return "Agent Memory"
        if "Multi" in question or "多 Agent" in question:
            return "多 Agent"
        if "MCP" in question:
            return "MCP Server"
        return "Phase4 Agent"

    def _code_query(self, question: str) -> str:
        if "Memory" in question or "记忆" in question:
            return "MemoryPolicy"
        if "Multi" in question or "多 Agent" in question:
            return "MultiAgentSupervisor"
        if "MCP" in question:
            return "search_docs"
        return "AgentRole"
