#!/usr/bin/env node

import { pathToFileURL } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import {
  callChatCompletion,
  resolveModelConfig,
  type ChatMessage,
  type ChatToolDefinition
} from "../model/chat_client.js";
import { loadEnvFiles } from "../model/env.js";

export interface RouteDemoOptions {
  request: string;
  origin: string;
  destination: string;
  city: string;
  cityd: string;
  mode: "driving" | "walking" | "transit" | "bicycling";
  includeFlight: boolean;
  departureAirport: string;
  departureAirportCoord: string;
  arrivalAirport: string;
  arrivalAirportCoord: string;
  raw: boolean;
  maxToolRounds: number;
}

interface ListedTool {
  name: string;
  description?: string;
  inputSchema?: {
    properties?: Record<string, unknown>;
    required?: string[];
  };
}

const DEFAULT_OPTIONS: RouteDemoOptions = {
  request:
    "我在北京西二旗地铁站，想去深圳北站，请比较高铁和飞机两类跨城出行方案。飞机方案只需要规划两端机场接驳，航班段不要编造。",
  // 北京西二旗地铁站 -> 深圳北站。跨城出行默认用公交/综合交通规划。
  origin: "116.306295,40.053034",
  destination: "114.029113,22.609767",
  city: "北京",
  cityd: "深圳",
  mode: "transit",
  includeFlight: true,
  departureAirport: "北京首都国际机场",
  departureAirportCoord: "116.615583,40.052657",
  arrivalAirport: "深圳宝安国际机场",
  arrivalAirportCoord: "113.814561,22.623291",
  raw: false,
  maxToolRounds: 5
};

const ROUTE_TOOL_BY_MODE: Record<RouteDemoOptions["mode"], string> = {
  driving: "maps_direction_driving",
  walking: "maps_direction_walking",
  transit: "maps_direction_transit_integrated",
  bicycling: "maps_bicycling"
};

export function parseArgs(argv: string[]): RouteDemoOptions {
  const options = { ...DEFAULT_OPTIONS };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--raw") {
      options.raw = true;
      continue;
    }
    if (arg === "--include-flight") {
      options.includeFlight = true;
      continue;
    }
    if (arg === "--no-flight") {
      options.includeFlight = false;
      continue;
    }

    const next = argv[i + 1];
    if (!next) {
      continue;
    }

    if (arg === "--origin") {
      options.origin = next;
      i += 1;
    } else if (arg === "--destination") {
      options.destination = next;
      i += 1;
    } else if (arg === "--city") {
      options.city = next;
      i += 1;
    } else if (arg === "--cityd") {
      options.cityd = next;
      i += 1;
    } else if (arg === "--mode" && isRouteMode(next)) {
      options.mode = next;
      i += 1;
    } else if (arg === "--request") {
      options.request = next;
      i += 1;
    } else if (arg === "--departure-airport") {
      options.departureAirport = next;
      i += 1;
    } else if (arg === "--departure-airport-coord") {
      options.departureAirportCoord = next;
      i += 1;
    } else if (arg === "--arrival-airport") {
      options.arrivalAirport = next;
      i += 1;
    } else if (arg === "--arrival-airport-coord") {
      options.arrivalAirportCoord = next;
      i += 1;
    } else if (arg === "--max-tool-rounds") {
      const parsed = Number.parseInt(next, 10);
      if (Number.isInteger(parsed) && parsed > 0) {
        options.maxToolRounds = parsed;
      }
      i += 1;
    }
  }

  return options;
}

export function extractTextContent(content: unknown): string {
  if (!Array.isArray(content)) {
    return JSON.stringify(content, null, 2);
  }

  return content
    .map((item) => {
      if (isTextContent(item)) {
        return item.text;
      }
      return JSON.stringify(item);
    })
    .join("\n")
    .trim();
}

export function buildRouteSummaryMessages(
  options: RouteDemoOptions,
  toolName: string,
  routeText: string
): ChatMessage[] {
  return [
    {
      role: "system",
      content:
        "你是一个严谨的出行规划助手。只能基于给定的 Amap MCP 返回结果整理答案，不能编造票价、班次余票、实时拥堵或不存在的路线。" +
        "如果原始结果缺少某项信息，就写“原始结果未提供”。输出中文 Markdown。"
    },
    {
      role: "user",
      content:
        "请把下面的高德地图 MCP 路线规划结果整理成适合人阅读的出行方案。\n\n" +
        `出发城市：${options.city}\n` +
        `到达城市：${options.cityd}\n` +
        `出行模式：${options.mode}\n` +
        `调用工具：${toolName}\n` +
        `起点坐标：${options.origin}\n` +
        `终点坐标：${options.destination}\n\n` +
        "输出格式要求：\n" +
        "1. 先给一段“推荐方案”摘要。\n" +
        "2. 再用表格列出每个可选方案的总耗时、主要交通链路、关键换乘点。\n" +
        "3. 再列出推荐方案的分段步骤。\n" +
        "4. 最后给出注意事项，明确哪些信息原始结果没有提供。\n\n" +
        `Amap MCP 原始结果：\n${routeText}`
    }
  ];
}

export function buildAmapAgentMessages(options: RouteDemoOptions): ChatMessage[] {
  return [
    {
      role: "system",
      content:
        "你是一个会使用 Amap MCP 工具的出行规划 Agent。你必须通过工具获取路线事实，再回答用户。" +
        "优先使用 maps_direction_transit_integrated 规划跨城公共交通/铁路出行；如用户要求飞机方案，只能用 Amap MCP 规划两端地面接驳。" +
        "Amap MCP 当前没有航班查询工具，不能查询或编造航班号、机票价格、起飞时间、落地时间、飞行时长、余票或机场安检耗时。" +
        "最终答案必须严格基于工具返回。不要编造票价、余票、实时延误、首末班车、建议预留时间或任何工具结果未提供的信息。" +
        "注意事项只能写“原始结果未提供”的信息，不要补充常识性建议。不要使用 emoji。最终答案用中文 Markdown，包含推荐摘要、方案表、分段步骤、注意事项。"
    },
    {
      role: "user",
      content:
        `${options.request}\n\n` +
        "可用的已知结构化信息如下，必要时可以直接用于工具调用：\n" +
        `- 起点坐标：${options.origin}\n` +
        `- 终点坐标：${options.destination}\n` +
        `- 出发城市：${options.city}\n` +
        `- 到达城市：${options.cityd}\n` +
        `- 建议出行模式：${options.mode}\n\n` +
        (options.includeFlight
          ? "用户需要飞机候选方案。Amap MCP 没有航班查询工具，请用工具获取这些地图事实：\n" +
            `- 高铁/公共交通整段方案：${options.origin} -> ${options.destination}\n` +
            `- 出发地到机场接驳：${options.origin} -> ${options.departureAirport}（${options.departureAirportCoord}）\n` +
            `- 到达机场到目的地接驳：${options.arrivalAirport}（${options.arrivalAirportCoord}） -> ${options.destination}\n` +
            "最终对比高铁方案和飞机候选方案时，飞机的航班段必须写“原始工具未提供，需要航班系统查询”。\n\n"
          : "") +
        "请先调用合适的 Amap MCP 工具，不要直接回答。"
    }
  ];
}

export function toOpenAiToolDefinitions(tools: ListedTool[]): ChatToolDefinition[] {
  const allowed = new Set([
    "maps_geo",
    "maps_regeocode",
    "maps_direction_driving",
    "maps_direction_walking",
    "maps_direction_transit_integrated",
    "maps_bicycling",
    "maps_distance",
    "maps_text_search"
  ]);

  return tools
    .filter((tool) => allowed.has(tool.name))
    .map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description ?? `Amap MCP tool: ${tool.name}`,
        parameters: normalizeToolParameters(tool.inputSchema)
      }
    }));
}

export function buildRouteArguments(tool: ListedTool, options: RouteDemoOptions): Record<string, string> {
  const properties = tool.inputSchema?.properties ?? {};
  const args: Record<string, string> = {};

  setIfSupported(args, properties, "origin", options.origin);
  setIfSupported(args, properties, "destination", options.destination);
  setIfSupported(args, properties, "city", options.city);
  setIfSupported(args, properties, "city1", options.city);
  setIfSupported(args, properties, "cityd", options.cityd);
  setIfSupported(args, properties, "city2", options.cityd);

  return Object.keys(args).length > 0
    ? args
    : {
        origin: options.origin,
        destination: options.destination
      };
}

export function pickRouteTool(tools: ListedTool[], mode: RouteDemoOptions["mode"]): ListedTool {
  const expectedName = ROUTE_TOOL_BY_MODE[mode];
  const tool = tools.find((item) => item.name === expectedName);
  if (!tool) {
    const names = tools.map((item) => item.name).sort().join(", ");
    throw new Error(`Amap route tool ${expectedName} was not found. Available tools: ${names}`);
  }
  return tool;
}

function setIfSupported(
  target: Record<string, string>,
  properties: Record<string, unknown>,
  key: string,
  value: string
) {
  if (Object.prototype.hasOwnProperty.call(properties, key)) {
    target[key] = value;
  }
}

function normalizeToolParameters(schema: ListedTool["inputSchema"]): Record<string, unknown> {
  if (!schema || typeof schema !== "object") {
    return {
      type: "object",
      properties: {},
      additionalProperties: true
    };
  }

  return {
    type: "object",
    properties: schema.properties ?? {},
    required: schema.required ?? []
  };
}

function isRouteMode(value: string): value is RouteDemoOptions["mode"] {
  return ["driving", "walking", "transit", "bicycling"].includes(value);
}

function isTextContent(value: unknown): value is { type: "text"; text: string } {
  return (
    typeof value === "object" &&
    value !== null &&
    "type" in value &&
    "text" in value &&
    (value as { type?: unknown }).type === "text" &&
    typeof (value as { text?: unknown }).text === "string"
  );
}

async function main() {
  loadEnvFiles();

  const apiKey = process.env.AMAP_MAPS_API_KEY;
  if (!apiKey) {
    console.log("AMAP_MAPS_API_KEY is not configured.");
    console.log("Add it to .env, then run:");
    console.log("npm run demo:amap");
    console.log("or:");
    console.log(
      "npm run demo:amap -- --mode transit --origin 116.306295,40.053034 --destination 114.029113,22.609767 --city 北京 --cityd 深圳"
    );
    return;
  }

  const options = parseArgs(process.argv.slice(2));
  const transport = new StdioClientTransport({
    command: "npx",
    args: ["-y", "@amap/amap-maps-mcp-server"],
    env: {
      ...process.env,
      AMAP_MAPS_API_KEY: apiKey
    }
  });
  const client = new Client({ name: "ai-agent-learn-amap-demo", version: "0.1.0" });

  try {
    await client.connect(transport);
    const list = await client.listTools();
    const tools = list.tools as ListedTool[];
    const openAiTools = toOpenAiToolDefinitions(tools);

    console.log("== Amap MCP tools ==");
    console.log(tools.map((tool) => tool.name).sort().join(", "));
    console.log("\n== Model-driven MCP tool loop ==");

    const modelConfig = resolveModelConfig();
    const messages = buildAmapAgentMessages(options);
    let totalUsage: Record<string, number> = {};

    for (let round = 1; round <= options.maxToolRounds; round += 1) {
      const response = await callChatCompletion(messages, modelConfig, {
        temperature: 0.2,
        maxTokens: 1400,
        tools: openAiTools,
        toolChoice: "auto"
      });
      totalUsage = mergeUsage(totalUsage, response.usage);

      if (!response.toolCalls || response.toolCalls.length === 0) {
        console.log("\n== Final answer ==\n");
        console.log(response.content);
        if (Object.keys(totalUsage).length > 0) {
          console.log("\n== Usage ==");
          console.log(JSON.stringify(totalUsage, null, 2));
        }
        return;
      }

      messages.push({
        role: "assistant",
        content: response.content || null,
        tool_calls: response.toolCalls
      });

      for (const toolCall of response.toolCalls) {
        const args = safeParseToolArguments(toolCall.function.arguments);
        console.log(
          `[round ${round}] model called ${toolCall.function.name} ${JSON.stringify(args)}`
        );

        const result = await client.callTool({
          name: toolCall.function.name,
          arguments: args
        });
        const toolResult = options.raw ? JSON.stringify(result.content, null, 2) : extractTextContent(result.content);

        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: toolResult
        });
      }
    }

    throw new Error(`model did not finish after ${options.maxToolRounds} MCP tool rounds`);
  } finally {
    await client.close();
  }
}

function safeParseToolArguments(raw: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

function mergeUsage(
  total: Record<string, number>,
  usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } | undefined
) {
  if (!usage) {
    return total;
  }

  return {
    prompt_tokens: (total.prompt_tokens ?? 0) + (usage.prompt_tokens ?? 0),
    completion_tokens: (total.completion_tokens ?? 0) + (usage.completion_tokens ?? 0),
    total_tokens: (total.total_tokens ?? 0) + (usage.total_tokens ?? 0)
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error: unknown) => {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    console.error(message);
    process.exit(1);
  });
}
