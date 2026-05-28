export type ChatRole = "system" | "user" | "assistant" | "tool";

export interface ChatToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatMessage {
  role: ChatRole;
  content: string | null;
  tool_call_id?: string;
  tool_calls?: ChatToolCall[];
}

export interface ChatToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface ModelConfig {
  provider: "deepseek" | "openai" | "custom";
  apiKey: string;
  baseUrl: string;
  model: string;
}

export interface ChatCompletionOptions {
  maxTokens?: number;
  temperature?: number;
  timeoutMs?: number;
  tools?: ChatToolDefinition[];
  toolChoice?: "auto" | "none";
}

export interface ChatCompletionResult {
  content: string;
  model: string;
  finishReason?: string;
  toolCalls?: ChatToolCall[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

interface ChatCompletionResponse {
  model?: string;
  choices?: Array<{
    finish_reason?: string;
    message?: {
      content?: string;
      tool_calls?: ChatToolCall[];
    };
  }>;
  usage?: ChatCompletionResult["usage"];
  error?: {
    message?: string;
  };
}

function normalizeModelName(model: string): string {
  if (model.startsWith("deepseek/")) {
    return model.slice("deepseek/".length);
  }
  if (model.startsWith("openai/")) {
    return model.slice("openai/".length);
  }
  return model;
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export function resolveModelConfig(env: NodeJS.ProcessEnv = process.env): ModelConfig {
  const deepseekKey = env.DEEPSEEK_API_KEY;
  const openaiKey = env.OPENAI_API_KEY;
  const genericKey = env.LLM_API_KEY;

  if (deepseekKey) {
    return {
      provider: "deepseek",
      apiKey: deepseekKey,
      baseUrl: trimTrailingSlash(env.DEEPSEEK_BASE_URL ?? env.LLM_BASE_URL ?? "https://api.deepseek.com"),
      model: normalizeModelName(env.LLM_MODEL ?? "deepseek-chat")
    };
  }

  if (openaiKey) {
    return {
      provider: "openai",
      apiKey: openaiKey,
      baseUrl: trimTrailingSlash(env.OPENAI_BASE_URL ?? env.LLM_BASE_URL ?? "https://api.openai.com/v1"),
      model: normalizeModelName(env.LLM_MODEL ?? "gpt-4o-mini")
    };
  }

  if (genericKey) {
    return {
      provider: "custom",
      apiKey: genericKey,
      baseUrl: trimTrailingSlash(env.LLM_BASE_URL ?? "https://api.deepseek.com"),
      model: normalizeModelName(env.LLM_MODEL ?? "deepseek-chat")
    };
  }

  throw new Error("missing model API key: set DEEPSEEK_API_KEY, OPENAI_API_KEY, or LLM_API_KEY in .env");
}

export async function callChatCompletion(
  messages: ChatMessage[],
  config: ModelConfig,
  options: ChatCompletionOptions = {}
): Promise<ChatCompletionResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? 60_000);

  try {
    const response = await fetch(`${config.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${config.apiKey}`
      },
      body: JSON.stringify({
        model: config.model,
        messages,
        temperature: options.temperature ?? 0.2,
        max_tokens: options.maxTokens ?? 900,
        ...(options.tools ? { tools: options.tools, tool_choice: options.toolChoice ?? "auto" } : {})
      }),
      signal: controller.signal
    });

    const body = (await response.json()) as ChatCompletionResponse;
    if (!response.ok) {
      throw new Error(body.error?.message ?? `model request failed with HTTP ${response.status}`);
    }

    const choice = body.choices?.[0];
    const content = choice?.message?.content?.trim() ?? "";
    const toolCalls = choice?.message?.tool_calls ?? [];
    if (!content && toolCalls.length === 0) {
      throw new Error("model response did not include assistant content");
    }

    return {
      content,
      model: body.model ?? config.model,
      finishReason: choice?.finish_reason,
      toolCalls,
      usage: body.usage
    };
  } finally {
    clearTimeout(timeout);
  }
}
