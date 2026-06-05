# 03 Memory System

本目录用于 Phase4 第三阶段：Agent 长期状态和跨会话记忆。

这里不重复 Phase2 的 memory-rag。Phase2 的重点是“从外部资料中检索知识”；这一阶段的重点是“Agent 如何记住和某个用户、项目、任务有关的长期上下文”。

```text
执行状态：一次任务执行到哪一步，适合 LangGraph checkpoint。
长期记忆：跨会话保留的用户偏好、实体信息、任务状态和历史决策。
```

## 当前实现

```text
03-memory-system/
├── short_term_state.py      # 本轮执行状态，不写入长期记忆
├── long_term_memory.py      # JSON 文件长期记忆存储和召回
├── memory_policy.py         # 记忆写入策略：该记什么、该拒绝什么
├── memory_agent_demo.py     # 带长期记忆的最小 Agent demo
└── tests/
    └── test_memory_system.py
```

配套文章：

```text
docs/phase-4/03-agent-memory-system.md
```

## 核心概念

### 1. 短期状态不是长期记忆

`ShortTermState` 只描述本轮执行：

```text
目标是什么？
已经执行了哪些步骤？
观察到了什么？
还有什么 pending action？
```

这些信息适合被 checkpoint 保存，用来暂停、恢复、回放一次工作流。但它不应该直接进入长期记忆，否则 Agent 会把临时推理过程也当成“关于用户的事实”。

### 2. 长期记忆需要类型

当前 demo 实现了三类长期记忆：

| 类型 | 例子 | 作用 |
|------|------|------|
| `preference` | 以后代码示例尽量用 Python | 调整回答风格 |
| `entity` | 当前项目叫 ai-agent-learn | 记住稳定实体 |
| `task` | Phase4 当前任务是实现 Agent Memory System | 跨会话恢复任务上下文 |

### 3. 不是所有输入都应该记住

`MemoryPolicy` 会拒绝明显敏感的信息，例如：

```text
API key
token
password
身份证
银行卡
```

这个阶段先不做复杂安全体系，但记忆系统从第一天就要有一个原则：**长期记忆不是垃圾桶，也不是日志库**。

## 运行方式

本模块只使用 Python 标准库，不需要安装额外依赖。

运行测试：

```bash
python3 -m unittest discover -s phase-4-advanced/03-memory-system/tests
```

当前测试覆盖：

- 稳定偏好可以写入长期记忆。
- 敏感内容不会写入长期记忆。
- 中文项目实体可以被识别为 `project_name`。
- `Phase4 当前任务是什么？` 这类疑问句不会被误写成新的 task memory。
- 同一类偏好更新时覆盖旧记忆，而不是无限追加。
- 按类型搜索不会误删其他类型记忆。
- 中文显式记忆会生成不同 `memory_id`。
- 短期状态不会混入长期记忆。
- Agent 后续回答会受到已召回记忆影响。

运行 demo：

```bash
python3 phase-4-advanced/03-memory-system/memory_agent_demo.py
```

也可以传入自定义多轮消息：

```bash
python3 phase-4-advanced/03-memory-system/memory_agent_demo.py \
  "以后回答我问题时，代码示例尽量用 Python。" \
  "记住：Phase4 当前任务是实现 Agent Memory System。" \
  "我接下来应该怎么学 Agent Memory？"
```

默认记忆文件写入：

```text
phase-4-advanced/03-memory-system/.memory/agent_memory.json
```

这是 demo 数据，不建议提交。

## 学习验收

学完这一节，应该能回答：

- RAG 和 Memory 的边界是什么？
- LangGraph checkpoint 为什么不等于长期记忆？
- 哪些用户输入值得写入长期记忆？
- 长期记忆为什么需要类型、置信度和更新时间？
- 当记忆冲突时，为什么应该更新旧记忆，而不是无限追加？
- Agent 回答前应该召回哪些记忆，而不是把所有记忆都塞进上下文？
