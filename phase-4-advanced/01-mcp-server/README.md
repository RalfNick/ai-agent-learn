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

如果要运行模型调用 demo，可以复用前面阶段已经配置好的 `.env`：

```bash
cp ../../phase-3-frameworks/01-framework-basics/01-langgraph-deep-dive/.env .env
npm run demo:model
```

当前学习阶段不真正发布 MCP Server，但可以做发布前检查：

```bash
npm run publish:check
```

这个命令只会执行 `npm pack --dry-run`，用于确认 npm 包里会包含哪些文件。`package.json` 里保留了 `"private": true`，避免学习阶段误执行真实发布。

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

## 模型调用 Demo

`src/demos/model_call_demo.ts` 演示的不是单纯的 “hello LLM”，而是一个更接近真实 Agent 的闭环：

```text
本地 MCP 工具实现取上下文
        ↓
search_docs / find_code_examples / read_benchmark_summary
        ↓
把受控上下文交给模型
        ↓
模型基于工程资料回答问题
```

默认问题是：

```text
Phase4 的 MCP Server 应该如何衔接前面的 Agentic RAG 学习？
```

也可以传入自己的问题：

```bash
npm run demo:model -- "这个 MCP Server 的安全边界应该怎么继续扩展？"
```

demo 会读取本目录 `.env`，并兜底读取前面阶段常用的 `.env` 路径。支持：

```text
DEEPSEEK_API_KEY + DEEPSEEK_BASE_URL
OPENAI_API_KEY + OPENAI_BASE_URL
LLM_API_KEY + LLM_BASE_URL
```

运行输出只展示模型名、provider 和 base_url，不打印 API key。

## 发布准备：只做 Dry Run

一个 MCP Server 真正进入可分发状态，至少要回答三个问题：

```text
别人怎么启动？
别人能看到哪些工具？
别人拿不到哪些东西？
```

本项目现在只做到发布前检查，不真正发布到 npm。检查项包括：

| 检查项 | 当前做法 |
|------|------|
| 构建产物 | `npm run build` 生成 `dist/` |
| 包内容 | `npm run publish:check` 查看 dry-run 文件列表 |
| 启动入口 | `bin.ai-agent-learn-mcp = dist/server.js` |
| 误发布保护 | `private: true` |
| Secret 保护 | `.env` 被 `.gitignore` 忽略，只提交 `.env.example` |

如果以后真的发布，需要先去掉 `private: true`，确认包名、README、License、版本号、工具安全边界，再执行真实 `npm publish`。这里先不做这一步，因为 Phase4 的重点是理解 MCP Server 的接口、分发形态和安全边界。

客户端配置可以长这样：

```json
{
  "mcpServers": {
    "ai-agent-learn": {
      "command": "npx",
      "args": ["-y", "ai-agent-learn-mcp-server"]
    }
  }
}
```

本地开发时更常用这种方式：

```json
{
  "mcpServers": {
    "ai-agent-learn-local": {
      "command": "node",
      "args": [
        "/Users/bytedance/ClaudeCode-Projects/ai-agent-learn/phase-4-advanced/01-mcp-server/dist/server.js"
      ]
    }
  }
}
```

## 公开 MCP 调用：Amap Maps 路线规划

只写自己的 MCP Server 容易停留在“工具注册”的视角。Phase4 还需要看外部 MCP 生态怎么接入，因为真实 Agent 往往会同时使用内部工具和外部服务。

mcp.so 上的 Amap Maps 是高德地图官方 MCP Server，包名是 `@amap/amap-maps-mcp-server`，工具包含：

```text
maps_geo
maps_regeocode
maps_weather
maps_direction_driving
maps_direction_walking
maps_direction_transit_integrated
maps_bicycling
maps_distance
maps_text_search
maps_around_search
```

高德官方文档目前推荐 Streamable HTTP，也支持 Node.js I/O。Node.js I/O 配置示例：

```json
{
  "mcpServers": {
    "amap-maps": {
      "command": "npx",
      "args": ["-y", "@amap/amap-maps-mcp-server"],
      "env": {
        "AMAP_MAPS_API_KEY": "your_amap_maps_api_key"
      }
    }
  }
}
```

本项目新增了一个路线规划 demo：

```bash
npm run demo:amap
```

默认场景是：

```text
北京西二旗地铁站 -> 深圳北站
```

跨城出行默认使用 `transit`，等价于：

```bash
npm run demo:amap -- --mode transit --origin 116.306295,40.053034 --destination 114.029113,22.609767 --city 北京 --cityd 深圳
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--mode` | `driving`、`walking`、`transit`、`bicycling` |
| `--origin` | 起点经纬度，格式 `lng,lat` |
| `--destination` | 终点经纬度，格式 `lng,lat` |
| `--city` | 公交规划等场景的起点城市，默认北京 |
| `--cityd` | 公交规划等场景的终点城市，默认深圳 |
| `--raw` | 同时打印 Amap MCP 原始返回，方便对照协议数据 |

如果 `.env` 里没有 `AMAP_MAPS_API_KEY`，demo 只会提示配置方式，不会发起外部调用。配置好 key 后，demo 会：

```text
启动 @amap/amap-maps-mcp-server
        ↓
listTools 查看公开 MCP 暴露的工具
        ↓
把 Amap MCP 工具 schema 交给模型
        ↓
模型决定调用 maps_direction_transit_integrated 等工具
        ↓
demo 执行模型发起的 MCP tool call
        ↓
把 tool result 回传给模型生成最终出行方案
```

默认输出会展示模型实际调用了哪个 MCP tool，例如：

```text
[round 1] model called maps_direction_transit_integrated {"origin":"116.306295,40.053034",...}
```

这才是更接近真实 Agent 的 MCP 使用方式：模型负责选择工具和参数，程序负责执行工具调用和回填结果。路线事实仍然来自 Amap MCP，最终答案要求只基于 tool result 生成。想看原始数据时再加 `--raw`：

```bash
npm run demo:amap -- --raw
```

这一步的学习重点不是“高德怎么用”，而是看清楚外部 MCP 接入的工程形态：

```text
内部 MCP：受控读取工程资料
外部 MCP：连接真实世界服务
Agent：在两类工具之间做规划、调用和结果整合
```

资料入口：

- mcp.so Amap Maps: https://mcp.so/server/amap-maps/amap
- 高德 MCP Server 快速接入: https://lbs.amap.com/api/mcp-server/gettingstarted

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
