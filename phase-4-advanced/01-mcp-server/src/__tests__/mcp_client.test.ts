import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

describe("MCP stdio server", () => {
  it("exposes tools to a real MCP client", async () => {
    const transport = new StdioClientTransport({
      command: process.execPath,
      args: ["dist/server.js"],
      cwd: process.cwd()
    });
    const client = new Client({ name: "ai-agent-learn-test-client", version: "0.1.0" });

    try {
      await client.connect(transport);
      const tools = await client.listTools();
      const names = tools.tools.map((tool) => tool.name);
      assert.deepEqual(
        names.sort(),
        ["find_code_examples", "read_benchmark_summary", "search_docs"].sort()
      );

      const result = await client.callTool({
        name: "read_benchmark_summary",
        arguments: { phase: "phase-3" }
      });
      assert.notEqual(result.isError, true);
      const content = result.content as Array<{ type: string; text?: string }>;
      const text = content
        .filter((item) => item.type === "text")
        .map((item) => item.text ?? "")
        .join("\n");
      assert.match(text, /agentic_rag_langgraph/);
    } finally {
      await client.close();
    }
  });
});
