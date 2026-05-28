import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { resolveModelConfig } from "../model/chat_client.js";
import { parseEnvFileContent } from "../model/env.js";

describe("model call demo helpers", () => {
  it("parses dotenv-style files without exposing comments", () => {
    const values = parseEnvFileContent(`
      # comment
      DEEPSEEK_API_KEY="secret"
      LLM_MODEL=deepseek/deepseek-chat
      EMPTY=
    `);

    assert.equal(values.DEEPSEEK_API_KEY, "secret");
    assert.equal(values.LLM_MODEL, "deepseek/deepseek-chat");
    assert.equal(values.EMPTY, "");
  });

  it("resolves DeepSeek config and normalizes LiteLLM-style model names", () => {
    const config = resolveModelConfig({
      DEEPSEEK_API_KEY: "test-key",
      LLM_MODEL: "deepseek/deepseek-chat"
    });

    assert.equal(config.provider, "deepseek");
    assert.equal(config.model, "deepseek-chat");
    assert.equal(config.baseUrl, "https://api.deepseek.com");
  });

  it("fails clearly when no model key is configured", () => {
    assert.throws(() => resolveModelConfig({}), /missing model API key/);
  });
});
