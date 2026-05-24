# 02 Agent Security

本目录用于 Phase4 第二阶段：围绕 MCP Server 和 Agent 工具调用设计安全边界。

计划重点：

- 路径 allowlist 和越权访问防护
- 工具参数校验
- Prompt 注入样例
- 输出脱敏
- 高风险工具的人类审批
- 只读工具到写工具的权限升级策略

第一阶段先不要在 MCP Server 中加入写文件、跑命令、触发 benchmark 等能力。等这里的安全 checklist 和攻防脚本完成后，再决定哪些能力可以开放。
