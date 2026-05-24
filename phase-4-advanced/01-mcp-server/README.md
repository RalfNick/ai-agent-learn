# 01 MCP Server

这个子项目是 Phase4 的第一个实战：把当前学习工程暴露成一个只读 MCP Server。

目标不是做一个“大而全”的工具服务，而是先把 MCP 的三个核心问题学清楚：

```text
Agent 能发现哪些工具？
Agent 能读取哪些资源？
Agent 不能越过哪些边界？
```

## 能力范围

第一版只读，不写文件、不执行 shell、不访问工程外目录。

### Tools

| Tool | 作用 |
|------|------|
| `search_docs` | 搜索 `docs/` 下的学习文章 |
| `find_code_examples` | 搜索各 phase 的示例代码 |
| `read_benchmark_summary` | 读取 Phase2 / Phase3 benchmark 汇总 |

### Resources

| Resource | 作用 |
|------|------|
| `docs://phase-1` | Phase1 文章索引 |
| `docs://phase-2` | Phase2 文章索引 |
| `docs://phase-3` | Phase3 文章索引 |
| `benchmark://phase-2` | Phase2 benchmark 汇总 |
| `benchmark://phase-3` | Phase3 benchmark 汇总 |

### Prompts

| Prompt | 作用 |
|------|------|
| `phase_review_prompt` | 辅助 review 某个学习阶段 |
| `article_outline_prompt` | 基于工程资料生成技术文章大纲 |

## 安装与运行

```bash
npm install
npm run build
npm test
```

启动 stdio server：

```bash
npm start
```

## 示例调用预期

`search_docs`：

```json
{
  "query": "Agentic RAG",
  "phase": "phase-3",
  "limit": 5
}
```

`find_code_examples`：

```json
{
  "query": "StateGraph",
  "phase": "phase-3",
  "limit": 5
}
```

`read_benchmark_summary`：

```json
{
  "phase": "phase-3"
}
```

## 安全边界

当前 server 的 allowlist 只包含：

```text
docs/
phase-1-fundamentals/
phase-2-rag/
phase-3-frameworks/
phase-4-advanced/
```

并且屏蔽：

```text
node_modules
dist
__pycache__
.git
.ruff_cache
.gradio
```

第一版刻意不提供写操作。后续如果要加入“触发 benchmark smoke test”或“生成文章草稿”等能力，必须先进入 `02-agent-security` 设计权限和审批。
