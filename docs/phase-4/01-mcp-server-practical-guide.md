# MCP 实战：把学习工程变成 Agent 可调用的工具服务

> Phase4 的第一篇文章。前面已经用 LangGraph 做了 Agentic RAG，现在开始补“工具协议”和“安全边界”这块能力。

Phase3 做完之后，Agent 已经不只是一次 LLM 调用。

它有状态，有路由，有 retry，有 repair，有拒答，也有 benchmark trace。

但这还不够。

企业级 Agent 最后一定要连接真实世界：

```text
读文档
查代码
看报表
访问数据库
调用内部系统
触发工作流
```

如果每个工具都靠临时函数拼接，系统很快会失控。MCP 要解决的就是这件事：用统一协议把工具、资源和提示模板暴露给 Agent。

---

## 一、这次不做玩具天气查询

很多 MCP 入门会从天气查询开始。

这个工程里不这么做。

这次的 MCP Server 直接服务当前学习工程：

```text
ai-agent-learn MCP Server
```

第一版只做三类只读能力：

| Tool | 作用 |
|------|------|
| `search_docs` | 搜索学习文章 |
| `find_code_examples` | 查找代码示例 |
| `read_benchmark_summary` | 读取 Phase2 / Phase3 benchmark |

这三个工具刚好对应前面几阶段的学习资产：

```text
文章是知识沉淀
代码是实现依据
benchmark 是验证结果
```

Agent 以后要 review 学习进度、生成文章大纲、解释某个阶段为什么这么设计，就不需要靠上下文里手动贴材料，而是可以通过 MCP 读取工程里的事实。

---

## 二、为什么第一版只读

第一版明确不做：

```text
不写文件
不执行 shell
不跑 benchmark
不修改文章
不访问工程外目录
```

这不是保守过头。

Agent 工具系统最危险的地方，不是“工具不够多”，而是“工具边界不清楚”。

如果一开始就给 Agent 文件写入、命令执行、外部路径访问，后面再补安全会非常难。正确顺序应该是：

```text
先只读
再校验
再加审批
最后才开放写操作
```

这也是 Phase4 的学习重点：不是让 Agent 什么都能做，而是让 Agent 在明确边界内做事。

---

## 三、MCP Server 的最小结构

当前子项目在：

```text
phase-4-advanced/01-mcp-server/
```

核心结构：

```text
src/server.ts
src/tools/search_docs.ts
src/tools/find_code_examples.ts
src/tools/read_benchmark_summary.ts
src/safety/path_guard.ts
src/__tests__/tools.test.ts
```

`server.ts` 只负责注册 MCP 能力：

```typescript
server.registerTool("search_docs", ...);
server.registerTool("find_code_examples", ...);
server.registerTool("read_benchmark_summary", ...);
```

真正的业务逻辑放在 `src/tools/` 里。

这样做有一个好处：工具函数可以直接单测，不必每次都启动 MCP 客户端。

---

## 四、三个工具分别解决什么

### 1. search_docs

输入：

```json
{
  "query": "Agentic RAG",
  "phase": "phase-3",
  "limit": 5
}
```

输出包括：

```text
文章路径
所属 phase
标题
匹配片段
```

这个工具解决的是“让 Agent 找到学习笔记里的依据”。

### 2. find_code_examples

输入：

```json
{
  "query": "StateGraph",
  "phase": "phase-3",
  "limit": 5
}
```

输出包括：

```text
脚本路径
语言
所属 phase
匹配片段
```

这个工具解决的是“让 Agent 不只讲概念，还能定位到真实代码”。

### 3. read_benchmark_summary

输入：

```json
{
  "phase": "phase-3"
}
```

读取：

```text
phase-2-rag/05-rag-benchmark/outputs/benchmark_summary.csv
phase-3-frameworks/02-agentic-rag-langgraph/outputs/agentic_rag_summary.csv
```

这个工具解决的是“让 Agent 的判断引用真实指标”。

比如 Phase3 的最新数字是：

```text
Faithfulness: 0.907 -> 0.980
平均延迟: 3269ms -> 5108ms
成本: $0.0296 -> $0.0443
LLM 调用: 60 -> 94
拒答: 6 次
```

---

## 五、安全边界放在哪里

第一版安全边界集中在：

```text
src/safety/path_guard.ts
```

当前 allowlist：

```text
docs/
phase-1-fundamentals/
phase-2-rag/
phase-3-frameworks/
phase-4-advanced/
```

屏蔽目录：

```text
node_modules
dist
__pycache__
.git
.ruff_cache
.gradio
```

参数也有限制：

```text
query 不能为空
query 最长 200 字符
limit 必须是 1 到 20 的整数
phase 必须是 phase-1 到 phase-4
```

这些规则看起来普通，但它们是 Agent 工具安全的地基。

---

## 六、怎么验证

进入目录：

```bash
cd phase-4-advanced/01-mcp-server
```

安装依赖：

```bash
npm install
```

运行测试：

```bash
npm test
```

构建：

```bash
npm run build
```

启动：

```bash
npm start
```

当前测试覆盖：

```text
搜索 Phase3 文档
查找 StateGraph 代码
读取 Phase3 benchmark
拒绝空 query
拒绝超长 query
拒绝非法 phase
拒绝 allowlist 外路径
拒绝工程外路径
```

这比“能启动 MCP Server”更重要。

因为工具系统一旦接给 Agent，就必须先证明边界有效。

---

## 七、下一步

这个 MCP Server 只是 Phase4 的入口。

下一步不是急着加更多工具，而是进入：

```text
phase-4-advanced/02-agent-security/
```

在那里补：

```text
Prompt 注入样例
路径越权攻击
输出脱敏
高风险工具审批
只读工具升级到写工具的规则
```

等安全边界清楚后，才适合考虑：

```text
触发 benchmark smoke test
生成文章草稿
更新学习复盘
写入长期记忆
```

这条路线更慢一点，但更接近企业级 Agent 开发。
