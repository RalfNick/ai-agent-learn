# 03 Memory System

本目录用于 Phase4 第三阶段：Agent 长期状态和跨会话记忆。

这里不重复 Phase2 的 memory-rag。Phase2 已经覆盖了记忆与检索结合，Phase4 要重点区分：

```text
执行状态：一次任务执行到哪一步，适合 LangGraph checkpoint。
长期记忆：跨会话保留的用户偏好、实体信息、任务状态和历史决策。
```

计划重点：

- 用户偏好记忆
- 实体记忆
- 任务状态记忆
- 记忆写入规则
- 记忆召回和遗忘策略
- 与 LangGraph checkpoint 的边界
