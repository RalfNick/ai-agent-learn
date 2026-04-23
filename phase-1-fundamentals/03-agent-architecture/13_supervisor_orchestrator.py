"""
13_supervisor_orchestrator.py — Supervisor 多 Agent 编排器

从 smolagents 的 ManagedAgent 和 langgraph 的 StateGraph 中提炼多 Agent 编排模式。
展示：Supervisor/Worker 模式、Agent 作为 Tool、任务分发与结果汇总。

对应文章第 5 章：多 Agent 编排
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


# ── 1. Worker Agent 抽象 ────────────────────────────────────────
# smolagents: ManagedAgent — 被管理的子 Agent，有 name/description
# langgraph:  StateGraph 的节点 — 每个节点是一个处理函数
# 共性：Worker 对外暴露统一接口（name + description + execute）

@dataclass
class WorkerAgent:
    """
    Worker Agent — Supervisor 可调度的最小单元。

    设计要点（从 smolagents ManagedAgent 提炼）：
    - name + description 让 Supervisor（LLM）理解每个 Worker 的能力
    - execute() 是统一接口，Supervisor 不关心内部实现
    - 这就是 smolagents "Agent 作为 Tool" 的核心思想
    """
    name: str
    description: str
    execute_fn: Callable[[str], str]
    tools: list[str] = field(default_factory=list)

    def execute(self, task: str) -> str:
        return self.execute_fn(task)


# ── 2. Supervisor 编排器 ────────────────────────────────────────
# smolagents: MultiStepAgent + managed_agents → Supervisor 自动选择子 Agent
# langgraph:  conditional_edges → 根据状态路由到不同节点
# awesome-llm-apps: Travel Planner 用 6 个 Agent 并行协作

class SupervisorOrchestrator:
    """
    Supervisor/Worker 编排模式。

    三种编排策略：
    1. LLM 路由（smolagents 方式）：Supervisor 是一个 Agent，由 LLM 决定调用哪个 Worker
    2. 规则路由（langgraph 方式）：根据状态条件路由到不同节点
    3. 全部执行（awesome-llm-apps Travel Planner 方式）：所有 Worker 并行执行，汇总结果
    """

    def __init__(self, workers: list[WorkerAgent]):
        self.workers = {w.name: w for w in workers}

    def route_by_rules(self, task: str, router_fn: Callable[[str], str]) -> str:
        """规则路由：根据任务内容选择 Worker"""
        worker_name = router_fn(task)
        if worker_name not in self.workers:
            return f"错误：未找到 Worker '{worker_name}'"
        worker = self.workers[worker_name]
        print(f"  📌 路由到: {worker.name}")
        return worker.execute(task)

    def execute_all(self, task: str) -> dict[str, str]:
        """全部执行：所有 Worker 处理同一任务，汇总结果"""
        results = {}
        for name, worker in self.workers.items():
            print(f"  🔄 {name} 执行中...")
            results[name] = worker.execute(task)
        return results

    def execute_pipeline(self, task: str, pipeline: list[str]) -> str:
        """流水线执行：按顺序传递，上一个的输出是下一个的输入"""
        current_input = task
        for worker_name in pipeline:
            worker = self.workers.get(worker_name)
            if not worker:
                return f"错误：未找到 Worker '{worker_name}'"
            print(f"  ➡️ {worker_name}: {current_input[:50]}...")
            current_input = worker.execute(current_input)
        return current_input

    def get_worker_descriptions(self) -> str:
        """生成 Worker 描述（注入 Supervisor 的 Prompt）"""
        lines = []
        for w in self.workers.values():
            tools_str = f" (工具: {', '.join(w.tools)})" if w.tools else ""
            lines.append(f"- {w.name}: {w.description}{tools_str}")
        return "\n".join(lines)


# ── 3. 演示 ─────────────────────────────────────────────────────

def create_demo_workers() -> list[WorkerAgent]:
    """创建演示用的 Worker Agent"""

    researcher = WorkerAgent(
        name="researcher",
        description="搜索和收集信息的研究员",
        tools=["web_search", "wiki_lookup"],
        execute_fn=lambda task: f"[研究结果] 关于'{task}'：AI Agent 市场预计 2026 年达到 500 亿美元规模",
    )

    analyst = WorkerAgent(
        name="analyst",
        description="分析数据并提供洞察的分析师",
        tools=["calculator", "chart_generator"],
        execute_fn=lambda task: f"[分析结论] {task} → 年增长率约 35%，主要驱动力是企业自动化需求",
    )

    writer = WorkerAgent(
        name="writer",
        description="撰写报告和总结的写手",
        execute_fn=lambda task: f"[报告] 综合分析：{task[:80]}... 建议关注 Agent 框架标准化趋势。",
    )

    return [researcher, analyst, writer]


def simple_router(task: str) -> str:
    """简单的规则路由"""
    task_lower = task.lower()
    if any(kw in task_lower for kw in ["搜索", "查找", "了解"]):
        return "researcher"
    elif any(kw in task_lower for kw in ["分析", "计算", "对比"]):
        return "analyst"
    else:
        return "writer"


if __name__ == "__main__":
    workers = create_demo_workers()
    supervisor = SupervisorOrchestrator(workers)

    print("=== Worker 描述（注入 Supervisor Prompt）===")
    print(supervisor.get_worker_descriptions())

    task = "分析 AI Agent 行业的发展趋势"

    # 策略 1：规则路由
    print(f"\n{'='*50}")
    print("策略 1：规则路由")
    result = supervisor.route_by_rules(task, simple_router)
    print(f"结果: {result}")

    # 策略 2：全部执行（类似 awesome-llm-apps Travel Planner）
    print(f"\n{'='*50}")
    print("策略 2：全部执行 + 汇总")
    results = supervisor.execute_all(task)
    for name, res in results.items():
        print(f"  {name}: {res}")

    # 策略 3：流水线
    print(f"\n{'='*50}")
    print("策略 3：流水线（研究 → 分析 → 撰写）")
    final = supervisor.execute_pipeline(task, ["researcher", "analyst", "writer"])
    print(f"最终输出: {final}")
