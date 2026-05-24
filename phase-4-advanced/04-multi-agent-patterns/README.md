# 04 Multi-Agent Patterns

本目录用于 Phase4 第四阶段：面向 Capstone 的多 Agent 协作模式。

不再泛泛写“多个角色聊天”的 demo，而是只实现和企业知识库 Agent 相关的模式：

- Supervisor
- Handoff
- Tool specialist
- Reviewer

核心问题：

```text
谁负责拆任务？
谁能调用工具？
谁检查答案质量？
失败时如何回退？
多 Agent 之间传递什么上下文？
```

实现时优先使用 LangGraph 显式路由，再和 CrewAI 的角色协作心智模型做对比。
