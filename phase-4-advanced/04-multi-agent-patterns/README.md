# 04 Multi-Agent Patterns

本目录用于 Phase4 第四阶段：面向 Capstone 的多 Agent 协作模式。

这里不做“多个角色聊天”的 demo，而是只实现和企业知识库 Agent 相关的工程模式：

- Supervisor：拆任务、决定谁处理。
- Handoff：明确移交合同，而不是一句自然语言“你来做”。
- Tool specialist：把文档、代码、benchmark 等能力收敛到专门角色。
- Reviewer：答案返回前做证据和边界检查。

## 当前实现

```text
04-multi-agent-patterns/
├── agents.py                    # role、report、review result、final result
├── handoff.py                   # HandoffPacket 和 SupervisorPlan
├── supervisor.py                # Supervisor、specialists、reviewer
├── multi_agent_demo.py          # 可运行 demo
└── tests/
    └── test_multi_agent_patterns.py
```

## 运行

本模块只使用 Python 标准库。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-4-advanced/04-multi-agent-patterns/tests
PYTHONDONTWRITEBYTECODE=1 python3 phase-4-advanced/04-multi-agent-patterns/multi_agent_demo.py
```

## 学习重点

学完这一节，应该能回答：

- 多 Agent 为什么不能只是角色 prompt？
- Supervisor 什么时候值得引入？
- Handoff 为什么需要结构化合同？
- Specialist 的工具边界怎么设计？
- Reviewer 检查什么，不能检查什么？
- 什么时候多 Agent 反而不划算？
