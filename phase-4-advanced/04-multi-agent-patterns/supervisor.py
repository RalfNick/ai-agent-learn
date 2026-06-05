from __future__ import annotations

from pathlib import Path

from agents import AgentRole, MultiAgentResult, ReviewResult, ReviewStatus, SpecialistReport
from handoff import HandoffPacket, SupervisorPlan


class DocResearchAgent:
    def handle(self, packet: HandoffPacket) -> SpecialistReport:
        return SpecialistReport(
            role=AgentRole.DOC_RESEARCHER,
            summary=(
                "DocResearchAgent: 检查文章是否围绕问题、架构、代码和取舍展开，"
                "避免只停留在学习笔记。"
            ),
            evidence=[
                "docs/phase-4/03-agent-memory-system.md",
                "docs/phase-4/README.md",
            ],
            risks=["文章需要引用真实代码路径和测试结果，不能只讲概念。"],
        )


class CodeAnalysisAgent:
    def handle(self, packet: HandoffPacket) -> SpecialistReport:
        return SpecialistReport(
            role=AgentRole.CODE_ANALYST,
            summary=(
                "CodeAnalysisAgent: 检查代码边界是否清楚，重点看 MemoryPolicy、"
                "JsonMemoryStore、tests 是否能证明行为。"
            ),
            evidence=[
                "phase-4-advanced/03-memory-system/memory_policy.py",
                "phase-4-advanced/03-memory-system/long_term_memory.py",
                "phase-4-advanced/03-memory-system/tests/test_memory_system.py",
            ],
            risks=["规则型 demo 容易被误解为生产系统，需要在文章里写明边界。"],
        )


class BenchmarkAgent:
    def handle(self, packet: HandoffPacket) -> SpecialistReport:
        return SpecialistReport(
            role=AgentRole.BENCHMARK_AGENT,
            summary=(
                "BenchmarkAgent: 检查是否有可复现实验或测试证据。当前 Memory 阶段"
                "不跑指标 benchmark，但用单元测试作为验收证据。"
            ),
            evidence=[
                "phase-4-advanced/03-memory-system/tests/test_memory_system.py",
            ],
            risks=["不要把 demo 输出当成系统质量证明，测试才是这一阶段的验收口径。"],
        )


class ReviewerAgent:
    def review(self, answer: str, evidence: list[str]) -> ReviewResult:
        comments: list[str] = []
        score = 0.45

        if not evidence:
            return ReviewResult(
                status=ReviewStatus.NEEDS_EVIDENCE,
                score=0.2,
                comments=["缺少 evidence，reviewer 不允许直接通过。"],
            )

        score += min(len(evidence), 5) * 0.08
        if "Agent" in answer or "Memory" in answer:
            score += 0.1
        if "risk" in answer.lower() or "风险" in answer:
            score += 0.1
        if score < 0.7:
            comments.append("结论有证据，但风险和边界还不够清楚。")
            return ReviewResult(ReviewStatus.NEEDS_REVISION, score, comments)

        comments.append("结论包含证据和边界说明，可以通过。")
        return ReviewResult(ReviewStatus.APPROVED, min(score, 0.95), comments)


class MultiAgentSupervisor:
    def __init__(self, project_root: Path | str | None = None) -> None:
        self.project_root = Path(project_root or Path.cwd())
        self.specialists = {
            AgentRole.DOC_RESEARCHER: DocResearchAgent(),
            AgentRole.CODE_ANALYST: CodeAnalysisAgent(),
            AgentRole.BENCHMARK_AGENT: BenchmarkAgent(),
        }
        self.reviewer = ReviewerAgent()

    def plan(self, question: str) -> SupervisorPlan:
        lower = question.lower()
        handoffs: list[HandoffPacket] = []

        if any(keyword in question for keyword in ["文章", "文档", "输出"]) or "article" in lower:
            handoffs.append(
                HandoffPacket(
                    target=AgentRole.DOC_RESEARCHER,
                    task="检查文章主线、结构、证据和读者价值。",
                    context={"question": question, "phase": "phase-4"},
                    required_outputs=["article_findings", "doc_refs"],
                    constraints=["只读分析，不直接修改文章。"],
                )
            )

        if any(keyword in question for keyword in ["代码", "实现", "架构"]) or "code" in lower:
            handoffs.append(
                HandoffPacket(
                    target=AgentRole.CODE_ANALYST,
                    task="检查代码架构、模块边界、测试覆盖和可运行性。",
                    context={"question": question, "phase": "phase-4"},
                    required_outputs=["file_refs", "risks", "test_gaps"],
                    constraints=["只读分析，不执行高风险操作。"],
                )
            )

        if any(keyword in question for keyword in ["benchmark", "指标", "测试", "证据"]):
            handoffs.append(
                HandoffPacket(
                    target=AgentRole.BENCHMARK_AGENT,
                    task="检查当前结论是否有测试、benchmark 或可复现证据支撑。",
                    context={"question": question, "phase": "phase-4"},
                    required_outputs=["evidence_refs", "acceptance_criteria"],
                    constraints=["不要把主观评价当成验收证据。"],
                )
            )

        if not handoffs:
            handoffs.append(
                HandoffPacket(
                    target=AgentRole.DOC_RESEARCHER,
                    task="先做问题澄清和资料定位。",
                    context={"question": question, "phase": "phase-4"},
                    required_outputs=["clarified_scope", "doc_refs"],
                    constraints=["不要臆造不存在的项目资料。"],
                )
            )

        return SupervisorPlan(question=question, handoffs=handoffs)

    def run(self, question: str) -> MultiAgentResult:
        trace = ["supervisor.plan"]
        plan = self.plan(question)
        reports: list[SpecialistReport] = []

        for packet in plan.handoffs:
            trace.append(f"handoff.{packet.target.value}")
            report = self.specialists[packet.target].handle(packet)
            reports.append(report)
            trace.append(f"specialist.{packet.target.value}.report")

        evidence = sorted({item for report in reports for item in report.evidence})
        answer = self._compose_answer(question, reports)
        review = self.reviewer.review(answer, evidence)
        trace.append("reviewer.review")

        return MultiAgentResult(
            answer=answer,
            handoffs=plan.handoffs,
            reports=reports,
            evidence=evidence,
            review=review,
            trace=trace,
        )

    def _compose_answer(self, question: str, reports: list[SpecialistReport]) -> str:
        sections = [f"Supervisor: 已将问题拆给 {len(reports)} 个 specialist。"]
        for report in reports:
            sections.append(report.summary)
            if report.risks:
                sections.append("风险: " + "；".join(report.risks))
        return "\n".join(sections)
