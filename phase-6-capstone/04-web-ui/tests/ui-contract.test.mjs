import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

import { demoAnswerResponse } from "../lib/demo-response.mjs";
import { buildFallbackAnswerResponse } from "../lib/fallback-response.mjs";
import { reviewLabel, reviewTone } from "../lib/format.mjs";

test("page exposes answer, sources, trace, and review status", () => {
  assert.equal(existsSync(new URL("../app/page.tsx", import.meta.url)), true);
  const page = readFileSync(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(page, /sources/i);
  assert.match(page, /trace/i);
  assert.match(page, /review_status|reviewStatus/i);
  assert.match(page, /api\/v1\/answer/);
});

test("demo response keeps the backend answer contract", () => {
  assert.equal(demoAnswerResponse.mode, "agentic_rag");
  assert.equal(demoAnswerResponse.review_status, "evidence_supported");
  assert.ok(demoAnswerResponse.answer.includes("trace"));
  assert.ok(demoAnswerResponse.sources.length >= 1);
  assert.ok(demoAnswerResponse.trace.length >= 3);
});

test("review formatting covers supported and abstained states", () => {
  assert.equal(reviewLabel("evidence_supported"), "Evidence Supported");
  assert.equal(reviewLabel("abstained"), "Abstained");
  assert.equal(reviewLabel("client_error"), "Client Error");
  assert.equal(reviewTone("evidence_supported"), "ok");
  assert.equal(reviewTone("abstained"), "warn");
  assert.equal(reviewTone("client_error"), "bad");
});

test("fallback response is an explicit client error, not the demo answer", () => {
  const response = buildFallbackAnswerResponse(
    "当前问题需要后端实时回答吗？",
    "web-session",
    "connect ECONNREFUSED",
  );

  assert.equal(response.question, "当前问题需要后端实时回答吗？");
  assert.equal(response.session_id, "web-session");
  assert.equal(response.mode, "client_error");
  assert.equal(response.review_status, "client_error");
  assert.deepEqual(response.sources, []);
  assert.ok(response.answer.includes("后端服务暂不可用"));
  assert.equal(response.answer.includes(demoAnswerResponse.answer), false);
  assert.equal(response.answer.includes("trace 展示：开发者要能调试路径"), false);
  assert.ok(response.trace.some((step) => step.step === "client.api_error"));
});
