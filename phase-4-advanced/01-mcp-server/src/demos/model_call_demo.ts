#!/usr/bin/env node

import { callChatCompletion, resolveModelConfig, type ChatMessage } from "../model/chat_client.js";
import { loadEnvFiles } from "../model/env.js";
import { findCodeExamples } from "../tools/find_code_examples.js";
import { readBenchmarkSummary } from "../tools/read_benchmark_summary.js";
import { searchDocs } from "../tools/search_docs.js";

function buildPromptContext(
  docs: Awaited<ReturnType<typeof searchDocs>>,
  code: Awaited<ReturnType<typeof findCodeExamples>>,
  benchmark: Awaited<ReturnType<typeof readBenchmarkSummary>>
): string {
  return JSON.stringify(
    {
      docs: docs.results.slice(0, 3),
      code: code.results.slice(0, 3),
      benchmark: benchmark.summaries
    },
    null,
    2
  );
}

async function main() {
  const question =
    process.argv.slice(2).join(" ").trim() ||
    "Phase4 的 MCP Server 应该如何衔接前面的 Agentic RAG 学习？";

  const loadedEnvFiles = loadEnvFiles();
  const config = resolveModelConfig();

  const [docs, code, benchmark] = await Promise.all([
    searchDocs({ query: "Agentic RAG MCP Server", phase: "phase-3", limit: 5 }),
    findCodeExamples({ query: "McpServer registerTool", phase: "phase-4", limit: 5 }),
    readBenchmarkSummary({ phase: "phase-3" })
  ]);

  const messages: ChatMessage[] = [
    {
      role: "system",
      content:
        "你是一个务实的 Agent 工程学习助手。回答必须基于给定 MCP 工具上下文，避免编造不存在的文件或指标。" +
        "本工程 MCP 工具名固定为 search_docs、find_code_examples、read_benchmark_summary，不要改写工具名。"
    },
    {
      role: "user",
      content:
        `问题：${question}\n\n` +
        "下面是 MCP 工具返回的工程上下文，请先归纳事实，再给出建议：\n\n" +
        buildPromptContext(docs, code, benchmark)
    }
  ];

  console.log("== MCP context ==");
  console.log(`docs=${docs.count}, code=${code.count}, benchmark=${benchmark.count}`);
  console.log(`env_files=${loadedEnvFiles.length}`);
  console.log(`model=${config.model}, provider=${config.provider}, base_url=${config.baseUrl}`);
  console.log("\n== Model answer ==\n");

  const response = await callChatCompletion(messages, config, {
    temperature: 0.2,
    maxTokens: 900
  });

  console.log(response.content);

  if (response.usage) {
    console.log("\n== Usage ==");
    console.log(JSON.stringify(response.usage, null, 2));
  }
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  console.error(message);
  process.exit(1);
});
