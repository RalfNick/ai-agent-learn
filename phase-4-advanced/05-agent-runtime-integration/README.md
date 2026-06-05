# Phase4 Runtime Integration

这个目录是 Phase4 的收口集成：把前面三个能力放进同一个最小 runtime。

```text
MCP-style tools  ->  只读访问 docs、code、benchmark
Memory           ->  跨会话保存用户偏好和任务状态
Multi-Agent      ->  supervisor 路由、specialist 分析、reviewer 审查
```

它不是生产 Agent，也不接真实 LLM。当前阶段故意使用确定性代码，目的是看清楚工程边界：

- Memory 什么时候读、什么时候写。
- Supervisor 如何把问题拆给不同 specialist。
- Project tools 如何提供只读证据。
- Reviewer 如何拒绝无 evidence 的结论。
- Trace 如何让一次 Agent 运行可复盘。

## 文件结构

```text
phase-4-advanced/05-agent-runtime-integration/
├── project_tools.py              # Python 版 MCP-style 只读工具
├── runtime.py                    # 集成 Memory、tools、multi-agent review
├── runtime_demo.py               # 可运行 demo
└── tests/test_runtime_integration.py
```

## 运行测试

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s phase-4-advanced/05-agent-runtime-integration/tests
```

## 运行 Demo

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase-4-advanced/05-agent-runtime-integration/runtime_demo.py
```

也可以换问题：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 phase-4-advanced/05-agent-runtime-integration/runtime_demo.py \
  --question "请结合 Phase4 的 MCP、Memory 和 Multi-Agent，说说下一步怎么进入 Phase5"
```

## 当前边界

第一版只做学习闭环，不做：

```text
真实 LLM tool calling
真实 MCP stdio client 调用
写文件工具
权限审批
FastAPI 服务化
LangGraph runtime
```

这些留到 Phase5 和 Capstone。Phase4 这里要证明的是：Agent 不是单个 prompt，而是一组可以组合、审查和复盘的工程能力。
