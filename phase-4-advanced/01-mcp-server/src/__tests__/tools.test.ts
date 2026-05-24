import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { resolve } from "node:path";
import { PROJECT_ROOT } from "../config.js";
import { ensureInsideAllowedRoots } from "../safety/path_guard.js";
import { findCodeExamples } from "../tools/find_code_examples.js";
import { readBenchmarkSummary } from "../tools/read_benchmark_summary.js";
import { searchDocs } from "../tools/search_docs.js";

describe("ai-agent-learn MCP tools", () => {
  it("searches Phase3 docs", async () => {
    const result = await searchDocs({ query: "Agentic RAG", phase: "phase-3", limit: 5 });
    assert.ok(result.count > 0);
    assert.ok(result.results.some((item) => item.path.includes("phase-3")));
  });

  it("finds LangGraph StateGraph code examples", async () => {
    const result = await findCodeExamples({ query: "StateGraph", phase: "phase-3", limit: 10 });
    assert.ok(result.count > 0);
    assert.ok(result.results.some((item) => item.path.includes("langgraph") || item.snippet.includes("StateGraph")));
  });

  it("reads Phase3 benchmark summary", async () => {
    const result = await readBenchmarkSummary({ phase: "phase-3" });
    assert.equal(result.count, 1);
    assert.equal(result.summaries[0].phase, "phase-3");
    assert.ok(result.summaries[0].rows.some((row) => row.mode === "agentic_rag_langgraph"));
  });

  it("rejects empty query", async () => {
    await assert.rejects(() => searchDocs({ query: "   " }), /query must not be empty/);
  });

  it("rejects overlong query", async () => {
    await assert.rejects(() => findCodeExamples({ query: "x".repeat(201) }), /200 characters/);
  });

  it("rejects unsupported phases", async () => {
    await assert.rejects(() => searchDocs({ query: "RAG", phase: "phase-9" }), /unsupported phase/);
  });

  it("rejects paths outside the allowlist", () => {
    assert.throws(() => ensureInsideAllowedRoots(resolve(PROJECT_ROOT, "ai-agent-learn-plan.md")), /allowlist/);
  });

  it("rejects paths outside the project root", () => {
    assert.throws(() => ensureInsideAllowedRoots("/tmp/outside.md"), /project root/);
  });
});
