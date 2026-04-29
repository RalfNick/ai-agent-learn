# AI Agent 系统性学习规划

## 概述

- **目标**：系统性学习 AI Agent 开发与架构设计，具备企业级 AI 建设能力
- **语言**：Python（Agent 核心逻辑）+ TypeScript（Web UI、MCP Server）
- **节奏**：业余学习，每天 1-2 小时，预计 1-2 个月完成
- **毕业项目**：企业知识库问答 Agent 系统
- **形式**：代码实践 + 文章输出（12-15 篇）
- **当前基础**：有编程能力，对 AI 有基础了解和使用经验

---

## 工程结构

```
ai-agent-learn/
├── docs/                               # 学习笔记与文章输出
│   └── phase-{1..6}/
├── phase-1-fundamentals/               # Agent 原理与手写实现
│   ├── 01-minimal-agent/               # 最小 Agent + 多轮对话 + 工具调用
│   ├── 02-smolagents-deep-dive/        # smolagents 框架深入（内部机制、工具生态、规划反思、安全沙箱）
│   └── 03-agent-architecture/          # Agent 架构进阶（执行引擎、工具系统、Prompt 引擎、多 Agent 编排）
├── phase-2-rag/                        # RAG 全链路
│   ├── 01-basic-rag/
│   ├── 02-advanced-chunking/
│   ├── 03-hybrid-search/
│   └── 04-rag-evaluation/
├── phase-3-frameworks/                 # 框架实战
│   ├── 01-langchain-basics/
│   ├── 02-langgraph-workflows/
│   ├── 03-crewai-multi-agent/
│   ├── 04-claude-agent-sdk/
│   └── 05-framework-comparison/
├── phase-4-advanced/                   # 进阶架构
│   ├── 01-memory-system/
│   ├── 02-multi-agent-patterns/
│   ├── 03-mcp-server/
│   └── 04-agent-security/
├── phase-5-production/                 # 部署与运维
│   ├── 01-fastapi-backend/
│   ├── 02-docker-deploy/
│   ├── 03-observability/
│   └── 04-testing-eval/
└── phase-6-capstone/                   # 毕业项目
    └── enterprise-agent-system/
```

---

## Phase 1：Agent 核心原理与手写实现（0.5周）

### 目标
不依赖任何框架，用纯 Python + LLM API 吃透 Agent 底层运行逻辑。

### 学习内容

**1.1 Agent 核心认知**
- Agent 的定义与本质：感知 → 推理 → 行动 → 反馈循环
- 与普通 LLM 对话的核心区别：自主决策、工具调用、多步推理
- 核心组件：规划（Planning）、记忆（Memory）、工具（Tools）、行动（Action）

**1.2 经典范式深入理解**
- ReAct（Reasoning + Acting）：思考-行动-观察循环
- Chain of Thought（CoT）：链式推理
- Reflexion：自我反思与改进
- Plan-and-Execute：先规划后执行

**1.3 极简 Agent 手写实现**
- 用纯 Python + OpenAI/Claude API 实现 ReAct Agent
- 实现工具调用系统（至少 3 种工具：搜索、计算、文件操作）
- 实现基础记忆（对话历史 + 摘要）
- 跑通完整的「思考 → 行动 → 观察 → 再思考」循环

### 实战项目
| 目录 | 内容 |
|------|------|
| `01-minimal-agent/` | 最小可运行 Agent，支持多轮对话 + 工具调用（4 脚本） |
| `02-smolagents-deep-dive/` | smolagents 框架深入：内部机制、工具生态、规划反思、安全沙箱（5 脚本） |
| `03-agent-architecture/` | Agent 架构进阶：执行引擎、工具系统、Prompt 引擎、多 Agent 编排（4 脚本） |

### 参考资源
- [smolagents](https://github.com/huggingface/smolagents)：~1000 行核心代码，最适合理解 Agent 本质
- [HuggingFace AI Agents Course](https://huggingface.co/learn/agents-course)
- [Microsoft AI Agents for Beginners](https://microsoft.github.io/ai-agents-for-beginners/)
- [Datawhale hello-agents](https://github.com/datawhalechina/hello-agents)（中文）
- [Awesome LLM Apps](https://github.com/Shubhamsaboo/awesome-llm-apps)：优秀 Agent 应用设计参考（Provider-Agnostic、多 Agent 编排、Agentic RAG）

### 文章输出
- ✅《从零手写一个 AI Agent：核心原理与 ReAct 实现》
- ✅《深入 Agent 架构设计：从源码理解框架的共性与本质》

---

## Phase 2：RAG 检索增强生成（2周）⭐ 重点

### 目标
掌握 RAG 全链路，从基础搭建到生产级优化。这是毕业项目（企业知识库问答系统）的核心基础。

### 学习内容

**2.1 RAG 核心架构**
- 文档处理：PDF/Word/Markdown 多格式加载与清洗
- 分块策略：固定大小、语义分块、递归分块、层级分块
- Embedding 模型：OpenAI text-embedding-3、BGE、Jina 等对比
- 向量数据库：Chroma（本地入门）、Milvus（生产级）、Pinecone（云端）

**2.2 检索优化**
- 基础向量检索 → 混合检索（向量 + BM25 关键词）
- Rerank 重排序：Cross-encoder 模型
- HyDE（假设文档嵌入）
- 多路检索与结果融合

**2.3 生成优化**
- 检索结果与 Prompt 融合策略
- 上下文窗口管理
- 幻觉抑制：引用溯源、置信度评估
- 多轮对话 RAG：历史对话管理

**2.4 评估体系（RAGAS）**
- Faithfulness（忠实度）、Answer Relevancy（答案相关性）
- Context Precision（上下文精确度）、Context Recall（上下文召回率）

### 实战项目
| 目录 | 内容 |
|------|------|
| `01-basic-rag/` | 本地知识库问答系统（Chroma + LangChain） |
| `02-advanced-chunking/` | 对比 3 种分块策略的效果差异 |
| `03-hybrid-search/` | 混合检索 + Rerank 实现 |
| `04-rag-evaluation/` | 用 RAGAS 评估并优化 RAG 系统，输出对比报告 |

### 参考资源
- [RAG 完整指南](https://nerdleveltech.com/guides/rag-systems)
- [LangChain RAG 教程](https://github.com/blackinkkkxi/RAG_langchain)（中文）
- [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)（中文）
- [记忆与检索](https://hello-agents.datawhale.cc/#/./chapter8/%E7%AC%AC%E5%85%AB%E7%AB%A0%20%E8%AE%B0%E5%BF%86%E4%B8%8E%E6%A3%80%E7%B4%A2?id=_831-rag%e7%9a%84%e5%9f%ba%e7%a1%80%e7%9f%a5%e8%af%86)
- [RAGAS 评估框架](https://docs.ragas.io/)

### 文章输出
- 《RAG 全链路实战：从文档处理到检索优化》
- 《RAG 优化实战：混合检索 + Rerank 让准确率翻倍》

---

## Phase 3：主流框架深度实战（1 周）

### 目标
掌握 LangChain/LangGraph/CrewAI/Claude Agent SDK 四大框架，理解各自适用场景。

### 学习内容

**3.1 LangChain 核心**
- 核心组件：Chains、Retrievers、Tools、Agents、Memory
- LCEL 表达式语言：链式调用、流式输出
- 实战：RAG + Agent 结合应用

**3.2 LangGraph 进阶⭐ 重点**
- 核心设计：状态图（StateGraph）、节点与边、条件路由
- 循环/分支逻辑、状态持久化（Checkpointing）
- 人机交互（Human-in-the-loop）
- 复杂工作流：多步骤任务拆解、容错重试、反思机制
- Plan-and-Execute 架构

> LangGraph 是 2026 年企业级 Agent 开发的主流选择（PyPI 月下载量 4700 万+），重点投入。

**3.3 CrewAI 多智能体**
- 核心概念：Agent（角色）、Task（任务）、Crew（团队）、Process（流程）
- 角色分工与协作模式
- 实战：多角色内容创作团队

**3.4 Claude Agent SDK**
- SDK 架构：Agent 循环、内置工具（文件读写、命令执行）
- Python + TypeScript 双语言支持
- 与 Anthropic API Tool Use 的关系
- 实战：构建一个 Claude 驱动的自动化 Agent

**3.5 框架横向对比（0.5 周）**
- 从同一个任务出发，用不同框架实现，对比开发体验、性能、可维护性

### 实战项目
| 目录 | 内容 |
|------|------|
| `01-langchain-basics/` | LangChain 多工具 Agent + RAG 结合 |
| `02-langgraph-workflows/` | LangGraph 复杂工作流（任务拆解 → 执行 → 反思 → 汇总） |
| `03-crewai-multi-agent/` | CrewAI 多角色协作系统（如：产品分析团队） |
| `04-claude-agent-sdk/` | Claude Agent SDK 自动化应用 |
| `05-framework-comparison/` | 同一任务的多框架实现对比报告 |

### 参考资源
- [LangGraph 101](https://github.com/langchain-ai/langgraph-101)
- [LangGraph 官方教程](https://langchain-ai.github.io/langgraph/tutorials/)
- [CrewAI 速成](https://github.com/alejandro-ao/crewai-crash-course)
- [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/agent-sdk)
- [Awesome LangGraph](https://github.com/von-development/awesome-LangGraph)

### 文章输出
- 《LangGraph 实战：构建复杂 AI Agent 工作流》
- 《主流 Agent 框架终极对比：LangChain vs LangGraph vs CrewAI》

---

## Phase 4：Agent 系统设计与进阶能力（1.5 周）

### 目标
掌握企业级 Agent 系统的核心进阶能力：记忆系统、多智能体架构、MCP、安全防护。

### 学习内容

**4.1 记忆系统深度设计**
- 记忆分层：短期（会话上下文）、长期（向量数据库）、实体记忆、摘要记忆
- 记忆管理：增删改查、遗忘机制、重要性排序
- 上下文工程（Context Engineering）：选择性上下文、滑动窗口、滚动摘要
- 多智能体共享记忆设计

**4.2 多智能体架构模式**
- 五大编排模式：
  - Supervisor/Worker（中心调度）
  - Hierarchical（层级决策）
  - Peer-to-Peer（去中心化协作）
  - Sequential Pipeline（流水线）
  - Marketplace/Auction（竞价分配）
- 任务分发、角色分工、通信协议、冲突解决
- 用 LangGraph 实现 Supervisor 模式

**4.3 MCP（Model Context Protocol）⭐ 重点**
- MCP 协议规范：Tools、Resources、Prompts 三大原语
- MCP Server 开发：TypeScript + Python
- MCP 生态：1000+ 现有 Server，与 Claude/Cursor/VS Code 集成
- 实战：开发自定义 MCP Server

> MCP 已成为 AI 工具集成的行业标准（2025.11 捐赠给 Linux Foundation，OpenAI/Anthropic/Block 联合推动）。

**4.4 Agent 安全与防护（1 周）**
- Prompt 注入攻击类型：直接注入、间接注入
- 六层防御架构：输入验证 → Prompt 隔离 → 工具授权 → 输出过滤 → 执行沙箱 → 运行时监控
- Guardrails 实现模式
- 数据安全与隐私保护

### 实战项目
| 目录 | 内容 |
|------|------|
| `01-memory-system/` | 完整记忆系统（短期 + 长期 + 实体 + 摘要） |
| `02-multi-agent-patterns/` | Supervisor 模式多智能体系统（4+ 角色） |
| `03-mcp-server/` | 自定义 MCP Server（如：本地文件管理、数据库查询） |
| `04-agent-security/` | Prompt 注入攻防演练 + Guardrails 实现 |

### 参考资源
- [MCP for Beginners](https://github.com/microsoft/mcp-for-beginners)
- [MCP 官方规范](https://modelcontextprotocol.io/)
- [MCP Server 开发指南](https://docs.anthropic.com/en/docs/agents-and-tools/mcp)
- [多智能体架构指南](https://www.agilesoftlabs.com/blog/2026/03/multi-agent-ai-systems-enterprise-guide)
- [Agent 安全防护](https://www.mintmcp.com/blog/prevention-detection-ai-agents)

### 文章输出
- 《AI Agent 记忆系统设计：从原理到实现》
- 《MCP 实战：开发你的第一个 MCP Server》
- 《多智能体架构模式：五大编排模式详解》

---

## Phase 5：部署、测试与运维（1 周）

### 目标
将 Agent 应用从本地 demo 推向生产环境，掌握部署、测试、监控全流程。

### 学习内容

**5.1 后端 API 开发**
- FastAPI 封装 Agent 服务：流式输出（SSE/WebSocket）、鉴权、限流
- 接口设计：RESTful API + 流式响应

**5.2 容器化部署**
- Dockerfile 编写与镜像优化
- Docker Compose 多服务编排（应用 + 向量数据库 + 缓存）
- 云平台部署（阿里云/AWS）

**5.3 可观测性**
- LangSmith / Langfuse：执行链路追踪、在线评估
- 日志收集与分析
- 成本监控与优化（Token 用量追踪、模型路由）

**5.4 测试与评估**
- Agent 功能测试：任务完成率、工具调用正确性
- 稳定性测试：并发、容错
- 效果评估：幻觉检测、准确率、鲁棒性
- CI/CD 集成自动化评估

**5.5 成本优化**
- Prompt 压缩、上下文窗口管理
- 模型路由（简单任务用便宜模型）
- Token 预算控制

### 实战项目
| 目录 | 内容 |
|------|------|
| `01-fastapi-backend/` | FastAPI 封装 RAG/Agent 服务，支持流式输出 |
| `02-docker-deploy/` | Docker Compose 编排 + 云服务器部署 |
| `03-observability/` | Langfuse 集成，实现链路追踪 + 成本监控 |
| `04-testing-eval/` | 完整测试套件 + 评估报告 |

### 参考资源
- [Langfuse 文档](https://langfuse.com/docs)
- [LangSmith 文档](https://docs.smith.langchain.com/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)

### 文章输出
- 《AI Agent 从本地到生产：部署全流程实战》
- 《Agent 可观测性实战：用 Langfuse 追踪每一步决策》

---

## Phase 6：毕业实战项目 — 企业知识库问答 Agent 系统（2周）⭐

### 目标
综合前 5 个阶段所学，完成一个企业级知识库问答 Agent 系统，作为核心作品。

### 系统架构

```
┌─────────────────────────────────────────────────┐
│                   Web UI (Next.js)               │
│         多轮对话 / 文档上传 / 权限管理            │
└──────────────────────┬──────────────────────────┘
                       │ SSE/WebSocket
┌──────────────────────▼──────────────────────────┐
│              API Layer (FastAPI)                  │
│        鉴权 / 限流 / 流式输出 / 错误处理          │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│         Agent Orchestration (LangGraph)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ RAG Agent│ │ QA Agent │ │ Document Agent   │ │
│  │ 检索+生成 │ │ 多轮对话  │ │ 文档处理+索引    │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│  ┌──────────────────────────────────────────────┐│
│  │         Memory System (短期+长期+摘要)        ││
│  └──────────────────────────────────────────────┘│
└──────────────────────┬──────────────────────────┘
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ Vector DB    │ │ MCP      │ │ Langfuse     │
│ (Milvus)     │ │ Servers  │ │ (可观测性)    │
│ 混合检索+Rerank│ │ 工具集成  │ │ 链路追踪     │
└──────────────┘ └──────────┘ └──────────────┘
```

### 核心功能
- 多格式文档上传与处理（PDF/Word/Markdown/TXT）
- RAG 检索：混合搜索（向量 + BM25）+ Rerank 重排序
- 多轮对话 + 完整记忆系统（短期 + 长期 + 摘要）
- LangGraph 工作流编排：文档处理 → 检索 → 生成 → 反思
- MCP Server 集成（自定义工具扩展）
- 权限管理 + 多租户支持
- Agent 安全防护（输入验证 + Prompt 隔离 + 输出过滤）

### 技术栈
- 后端：Python + FastAPI + LangGraph + LangChain
- 前端：TypeScript + Next.js + React
- 向量数据库：Milvus（生产级）/ Chroma（开发环境）
- MCP Server：TypeScript
- 可观测性：Langfuse
- 部署：Docker Compose + 云服务器

### 交付物
- GitHub 开源仓库（高质量 README + 架构文档 + 部署文档）
- 系列技术文章（3-5 篇，覆盖架构设计、RAG 优化、部署实战）
- 可在线访问的 Demo
- RAGAS 评估报告（优化前后对比）

---

## 时间线总览

| 阶段 | 内容 | 预估时间 | 核心产出 |
|------|------|----------|----------|
| Phase 1 | Agent 原理与手写实现 | 0.5 周 | ReAct Agent + 工具系统 |
| Phase 2 | RAG 全链路 ⭐ | 2 周 | 知识库问答系统 + RAGAS 评估报告 |
| Phase 3 | 框架深度实战 | 1 周 | 4 框架实战 + 对比报告 |
| Phase 4 | 进阶架构设计 | 1.5 周 | 记忆系统 + MCP Server + 多智能体 |
| Phase 5 | 部署测试运维 | 1 周 | 生产级部署 + 监控 |
| Phase 6 | 毕业项目 ⭐ | 2 周 | 企业知识库问答 Agent 系统 |
| **总计** | | **8周（约2个月）** | |

---

## 关键学习资源

### 课程
- [HuggingFace AI Agents Course](https://huggingface.co/learn/agents-course)
- [Microsoft AI Agents for Beginners](https://microsoft.github.io/ai-agents-for-beginners/)
- [Datawhale hello-agents](https://hello-agents.datawhale.cc/)（中文）

### 框架文档
- [LangGraph](https://langchain-ai.github.io/langgraph/) | [CrewAI](https://docs.crewai.com/) | [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/agent-sdk) | [MCP](https://modelcontextprotocol.io/)

### 开源项目
- [smolagents](https://github.com/huggingface/smolagents) — 理解 Agent 原理
- [LangGraph 101](https://github.com/langchain-ai/langgraph-101) — LangGraph 入门
- [Awesome LLM Apps](https://github.com/Shubhamsaboo/awesome-llm-apps) — 优秀 Agent 应用设计参考
- [MCP for Beginners](https://github.com/microsoft/mcp-for-beginners) — MCP 入门

### 工具
- [Langfuse](https://langfuse.com/)（可观测性）| [RAGAS](https://docs.ragas.io/)（RAG 评估）| [Chroma](https://www.trychroma.com/)（向量数据库）

---

## 验证标准

- 每个 Phase 的实战项目必须可运行
- RAG 系统 RAGAS 评估分数：Faithfulness > 0.8, Relevancy > 0.7
- 框架对比需有量化数据（开发时间、Token 消耗、任务完成率）
- MCP Server 需能在 Claude Code / Cursor 中正常调用
- 毕业项目需 Docker 一键部署 + 可公网访问

## 文章输出计划（12-15 篇）

| 阶段 | 文章 |
|------|------|
| Phase 1 | ✅《从零手写 AI Agent》✅《深入 Agent 架构设计》 |
| Phase 2 | 《RAG 全链路实战》《混合检索 + Rerank 优化》 |
| Phase 3 | 《LangGraph 工作流实战》《Agent 框架终极对比》 |
| Phase 4 | 《Agent 记忆系统设计》《MCP Server 开发实战》《多智能体架构模式》 |
| Phase 5 | 《Agent 部署全流程》《Langfuse 可观测性实战》 |
| Phase 6 | 《企业知识库问答系统：架构设计》《RAG 优化实战报告》《从 0 到 1 开源一个企业级 Agent 项目》 |
