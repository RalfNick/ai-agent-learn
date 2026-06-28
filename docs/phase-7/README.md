# Phase7：成熟开源 Agent 框架研究

Phase7 不再从零实现 Agent，而是把前六个阶段的经验拿去读成熟开源工程。

这一阶段研究两个项目：

| 项目 | baseline | 第一轮定位 |
| --- | --- | --- |
| OpenClaw | `a21144d8` | local-first Gateway、多渠道、节点、插件 SDK、安全与沙箱模型 |
| Hermes Agent | `a4fa148` | 远端常驻个人 Agent、记忆、skills、cron、MCP、飞书/Lark 工作流 |

## 文章目录

三篇主体文章分别对应原计划的三个部分：整体特点与应用、架构研究、个人工作流实践。`00` 是资料档案，不算主体文章。

| 文件 | 主题 |
| --- | --- |
| `00-openclaw-hermes-source-dossier.md` | OpenClaw 与 Hermes 官方文档、社区博客和播客资料档案 |
| `01-open-source-agent-framework-landscape.md` | 整体特点、应用场景和选型边界 |
| `02-openclaw-hermes-architecture-study.md` | OpenClaw 与 Hermes 的架构主线源码研读 |
| `03-hermes-feishu-personal-workflow.md` | 基于 Hermes、VPS 和飞书 WebSocket 的个人工作流实践 |

## 实践目录

配套实践材料在：

```text
phase-7-open-source-frameworks/01-hermes-feishu-workflow/
```

该目录只放无密钥模板、部署清单、验收说明和实践记录，不提交真实 API key、Feishu App Secret、用户 ID 或 VPS 地址。

## 与前六个阶段的关系

Phase7 不是替代前面的学习工程，而是做一次反向校准：

```text
Phase1：手写 Agent loop，理解 ReAct 和 tool calling
Phase2：RAG、检索、评估
Phase3：LangGraph、CrewAI、SDK 横向实践
Phase4：MCP、Memory、多 Agent、安全边界
Phase5：FastAPI、Docker、observability、eval
Phase6：企业知识库 Agent capstone
Phase7：用成熟开源项目检查这些模块在真实工程里如何组织
```

这次重点不是“哪个项目更强”，而是回答：

```text
成熟 Agent 工程如何处理入口通道、长生命周期、工具边界、记忆、自动化、安全和部署？
```
