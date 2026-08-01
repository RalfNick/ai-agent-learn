# Skill Analysis：Agent Skills 工程研究

这个目录研究的不是“有哪些好用的提示词”，而是 Skills 如何进入 Agent 上下文、如何约束执行流程、如何跨 Harness 适配，以及如何用行为评测验证它们是否真的有效。

## 文章目录

| 文件 | 主题 |
| --- | --- |
| [`01-superpowers-engineering-methodology.md`](./01-superpowers-engineering-methodology.md) | 从安装使用、bootstrap、14 个 Skills、开发生命周期和 Quorum 评测，完整拆解 Superpowers |

## 研究 baseline

| 项目 | baseline | 用途 |
| --- | --- | --- |
| [obra/superpowers](https://github.com/obra/superpowers/tree/44c9b2d6e889982ac18c27d05a19fefe335194e1) | `44c9b2d`，插件版本 `v6.2.0` | Skills、插件清单、bootstrap、Harness 适配、发布说明和静态测试 |
| [prime-radiant-inc/superpowers-evals](https://github.com/prime-radiant-inc/superpowers-evals/tree/8ed824a04d3e98c5789438fbdd0794399405776d) | `8ed824a` | Quorum 评测方法、Codex baseline、效率实验和失败案例 |

本文以 Codex 为主要使用平台，同时对照 Claude Code、Cursor、OpenCode、Pi 和 Gemini 的适配机制。所有效果数字都保留版本、样本和限制；项目自己的 release note 与 eval 结果视为一手工程证据，而不是无条件推广到所有模型和任务的普遍结论。
