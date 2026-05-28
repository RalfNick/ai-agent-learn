import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { PROJECT_ROOT } from "../config.js";

export const ENV_FILE_CANDIDATES = [
  resolve(PROJECT_ROOT, "phase-4-advanced/01-mcp-server/.env"),
  resolve(PROJECT_ROOT, ".env"),
  resolve(PROJECT_ROOT, "phase-3-frameworks/01-framework-basics/01-langgraph-deep-dive/.env"),
  resolve(PROJECT_ROOT, "phase-3-frameworks/01-framework-basics/02-crewai-multi-agent/.env"),
  resolve(PROJECT_ROOT, "phase-2-rag/04-rag-evaluation/.env"),
  resolve(PROJECT_ROOT, "phase-2-rag/01-basic-rag/.env")
];

export function parseEnvFileContent(content: string): Record<string, string> {
  const values: Record<string, string> = {};

  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }

    const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if (!match) {
      continue;
    }

    const [, key, rawValue] = match;
    let value = rawValue.trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    values[key] = value;
  }

  return values;
}

export function loadEnvFiles(
  candidates: string[] = ENV_FILE_CANDIDATES,
  target: NodeJS.ProcessEnv = process.env
): string[] {
  const loaded: string[] = [];

  for (const file of candidates) {
    if (!existsSync(file)) {
      continue;
    }

    const values = parseEnvFileContent(readFileSync(file, "utf8"));
    for (const [key, value] of Object.entries(values)) {
      target[key] ??= value;
    }
    loaded.push(file);
  }

  return loaded;
}
