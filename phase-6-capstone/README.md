# Phase6 Capstone

Phase6 的目标是做一个完整但不过度膨胀的企业知识库 Agent 系统。

它不是重新学习一个新框架，而是组合前五个阶段：

- Phase2：真实 RAG benchmark、hybrid search、rerank、faithfulness。
- Phase3：LangGraph Agentic RAG、rewrite、repair、abstain、trace。
- Phase4：MCP-style tools、memory、多 Agent reviewer。
- Phase5：FastAPI、Docker、observability、API eval。

## 推荐实现顺序

```text
phase-6-capstone/
├── 01-backend-skeleton/       # FastAPI + shared schemas + health
├── 02-knowledge-ingestion/    # 文档导入、chunk、index、hybrid retrieval
├── 03-agentic-qa-runtime/     # LangGraph Agentic RAG 主流程
├── 04-web-ui/                 # Next.js chat + sources + trace view
└── 05-release-eval/           # Docker compose + eval replay + observability
```

## 验收标准

- 能导入一批真实 Markdown/PDF 文档。
- 能回答知识库问题，并展示引用来源。
- 能在资料不足时拒答。
- 能显示每次回答的 trace、工具调用、延迟和 review 状态。
- 能运行一组 golden set eval cases，输出 pass rate。
- 能通过 Docker Compose 一键启动。

## 当前状态

- [x] Phase6 总体设计
- [x] Backend skeleton
- [x] Knowledge ingestion
- [x] Agentic QA runtime
- [x] Web UI
- [x] Release eval
