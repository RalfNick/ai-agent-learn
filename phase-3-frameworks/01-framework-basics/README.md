# Phase3 Framework Basics

这个目录是 Phase3 的框架基础入口，保留 LangChain、LangGraph、CrewAI、Claude/Codex 类工具循环和框架对比 demo。

## 推荐学习顺序

```text
01-framework-basics/
├── 00-langchain-foundations/
├── 01-langgraph-deep-dive/
├── 02-crewai-multi-agent/
├── 03-claude-agent-sdk/
└── 04-framework-showdown/
```

建议先从 LangChain 基础开始：

```bash
cd 00-langchain-foundations
python3 01_lcel_and_tools.py
```

再进入 LangGraph：

```bash
cd 01-langgraph-deep-dive
python3 01_state_graph_basics.py
```

再读：

```text
docs/phase-3/00-langchain-to-langgraph-foundations.md
```

再进入：

```text
phase-3-frameworks/02-agentic-rag-langgraph/
```

## 各目录定位

- `00-langchain-foundations/`：LCEL、RunnableParallel、Tool schema 的最小可运行示例
- `01-langgraph-deep-dive/`：StateGraph、Human-in-the-loop、Plan-and-execute、Agentic RAG、持久化记忆
- `02-crewai-multi-agent/`：CrewAI 的角色协作、多 Agent 分工和层级委派
- `03-claude-agent-sdk/`：工具循环、Guardrails、handoff、自主代码操作
- `04-framework-showdown/`：同一任务下对比 LangGraph、CrewAI、Claude/Codex 类 SDK

## Smoke Test 建议

基础学习建议优先跑这些脚本：

```bash
python3 00-langchain-foundations/01_lcel_and_tools.py
python3 01-langgraph-deep-dive/01_state_graph_basics.py
python3 01-langgraph-deep-dive/02_human_in_the_loop.py
python3 01-langgraph-deep-dive/03_plan_and_execute.py
python3 01-langgraph-deep-dive/04_agentic_rag.py
python3 01-langgraph-deep-dive/05_persistence_memory.py
python3 02-crewai-multi-agent/01_crew_basics.py
python3 02-crewai-multi-agent/03_product_analysis_crew.py
python3 03-claude-agent-sdk/02_guardrails_handoffs.py
python3 03-claude-agent-sdk/03_autonomous_coder.py
python3 04-framework-showdown/01_task_definition.py
python3 04-framework-showdown/05_comparison_report.py
```

`02-crewai-multi-agent/02_hierarchical_delegation.py` 是重型观察脚本：可以跑通，但会展开大量隐式委派、代码生成和审查过程，运行时间很长，不建议作为日常 smoke test。

`03-claude-agent-sdk/01_agent_loop.py` 和 `04-framework-showdown/04_claude_sdk_solution.py` 需要 `ANTHROPIC_API_KEY` 才能完整运行；没有 key 时会展示降级说明。

Phase3 当前主线是：

```text
框架基础采样 -> LangGraph 深入 -> Agentic RAG with LangGraph -> 框架对比
```
