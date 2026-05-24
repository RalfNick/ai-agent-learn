#!/usr/bin/env node

import { join } from "node:path";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import * as z from "zod/v4";
import { DOCS_ROOT } from "./config.js";
import { ensureInsideAllowedRoots, toProjectRelative } from "./safety/path_guard.js";
import { asJsonText, readTextFile } from "./tools/common.js";
import { findCodeExamples } from "./tools/find_code_examples.js";
import { readBenchmarkSummary } from "./tools/read_benchmark_summary.js";
import { searchDocs } from "./tools/search_docs.js";

const phaseSchema = z.enum(["phase-1", "phase-2", "phase-3", "phase-4"]).optional();
const limitSchema = z.number().int().min(1).max(20).optional();

export function createServer(): McpServer {
  const server = new McpServer(
    {
      name: "ai-agent-learn-mcp-server",
      version: "0.1.0"
    },
    {
      instructions:
        "Read-only learning workspace server. Use it to search docs, find code examples, and inspect benchmark summaries. It never writes files or executes shell commands."
    }
  );

  server.registerTool(
    "search_docs",
    {
      description: "Search Markdown learning articles under docs/ and return matching paths and snippets.",
      inputSchema: z.object({
        query: z.string().min(1).max(200),
        phase: phaseSchema,
        limit: limitSchema
      })
    },
    async (args) => asJsonText(await searchDocs(args))
  );

  server.registerTool(
    "find_code_examples",
    {
      description: "Search read-only code examples across phase directories and return matching scripts.",
      inputSchema: z.object({
        query: z.string().min(1).max(200),
        phase: phaseSchema,
        limit: limitSchema
      })
    },
    async (args) => asJsonText(await findCodeExamples(args))
  );

  server.registerTool(
    "read_benchmark_summary",
    {
      description: "Read Phase2 or Phase3 benchmark summary CSV files as structured JSON.",
      inputSchema: z.object({
        phase: z.enum(["phase-2", "phase-3", "all"]).optional()
      })
    },
    async (args) => asJsonText(await readBenchmarkSummary(args))
  );

  for (const phase of ["phase-1", "phase-2", "phase-3"] as const) {
    server.registerResource(
      `docs-${phase}`,
      `docs://${phase}`,
      {
        title: `${phase} article index`,
        description: `Markdown article index for ${phase}.`,
        mimeType: "application/json"
      },
      async (uri) => ({
        contents: [
          {
            uri: uri.href,
            mimeType: "application/json",
            text: JSON.stringify(await loadDocsIndex(phase), null, 2)
          }
        ]
      })
    );
  }

  for (const phase of ["phase-2", "phase-3"] as const) {
    server.registerResource(
      `benchmark-${phase}`,
      `benchmark://${phase}`,
      {
        title: `${phase} benchmark summary`,
        description: `Benchmark summary rows for ${phase}.`,
        mimeType: "application/json"
      },
      async (uri) => ({
        contents: [
          {
            uri: uri.href,
            mimeType: "application/json",
            text: JSON.stringify(await readBenchmarkSummary({ phase }), null, 2)
          }
        ]
      })
    );
  }

  server.registerPrompt(
    "phase_review_prompt",
    {
      title: "Phase review prompt",
      description: "Review a learning phase using docs, code examples, and benchmark results.",
      argsSchema: {
        phase: z.enum(["phase-1", "phase-2", "phase-3", "phase-4"])
      }
    },
    ({ phase }) => ({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text:
              `请 review ${phase} 的学习状态。先调用 search_docs 和 find_code_examples，` +
              "如果是 phase-2 或 phase-3，再调用 read_benchmark_summary。输出：当前产物、能力达标情况、缺口和下一步建议。"
          }
        }
      ]
    })
  );

  server.registerPrompt(
    "article_outline_prompt",
    {
      title: "Technical article outline prompt",
      description: "Create a technical article outline grounded in the workspace materials.",
      argsSchema: {
        topic: z.string().min(1).max(120),
        phase: phaseSchema
      }
    },
    ({ topic, phase }) => ({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text:
              `请围绕“${topic}”输出一篇公众号技术文章大纲。` +
              `${phase ? `优先搜索 ${phase} 的文章和代码。` : "先搜索相关文档和代码。"} ` +
              "要求包含真实工程路径、关键代码点、图示建议、读者能带走的工程判断。"
          }
        }
      ]
    })
  );

  return server;
}

async function loadDocsIndex(phase: "phase-1" | "phase-2" | "phase-3") {
  const dir = ensureInsideAllowedRoots(join(DOCS_ROOT, phase));
  const { readdir } = await import("node:fs/promises");
  const entries = await readdir(dir);
  const files = entries.filter((entry) => entry.endsWith(".md")).sort();
  return Promise.all(
    files.map(async (file) => {
      const path = join(dir, file);
      const text = await readTextFile(path);
      return {
        uri: `file://${path}`,
        path: toProjectRelative(path),
        title: text.match(/^#\s+(.+)$/m)?.[1] ?? file
      };
    })
  );
}

async function main() {
  const server = createServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
});
