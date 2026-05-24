import { readdir, readFile, stat } from "node:fs/promises";
import { basename, extname, join } from "node:path";
import { CODE_ROOTS, DOCS_ROOT, PHASES, type Phase } from "../config.js";
import { ensureInsideAllowedRoots, isBlockedPath, toProjectRelative } from "../safety/path_guard.js";

export interface SearchHit {
  path: string;
  phase: Phase | "unknown";
  title: string;
  score: number;
  snippet: string;
}

export function normalizePhase(phase?: string): Phase | undefined {
  if (!phase) {
    return undefined;
  }
  if ((PHASES as string[]).includes(phase)) {
    return phase as Phase;
  }
  throw new Error(`unsupported phase: ${phase}`);
}

export function phaseFromPath(path: string): Phase | "unknown" {
  const rel = toProjectRelative(path);
  const match = rel.match(/(?:docs\/|^)(phase-[1-4])\b|^phase-([1-4])-/);
  if (!match) {
    return "unknown";
  }
  return (match[1] ?? `phase-${match[2]}`) as Phase;
}

export async function readTextFile(path: string): Promise<string> {
  ensureInsideAllowedRoots(path);
  return readFile(path, "utf-8");
}

export async function walkFiles(root: string, shouldInclude: (path: string) => boolean): Promise<string[]> {
  const absoluteRoot = ensureInsideAllowedRoots(root);
  const result: string[] = [];
  async function visit(current: string): Promise<void> {
    if (isBlockedPath(current)) {
      return;
    }
    const info = await stat(current);
    if (info.isDirectory()) {
      const entries = await readdir(current);
      for (const entry of entries) {
        await visit(join(current, entry));
      }
      return;
    }
    if (info.isFile() && shouldInclude(current)) {
      result.push(current);
    }
  }
  await visit(absoluteRoot);
  return result;
}

export function docsRootsForPhase(phase?: Phase): string[] {
  return phase ? [join(DOCS_ROOT, phase)] : [DOCS_ROOT];
}

export function codeRootsForPhase(phase?: Phase): string[] {
  if (!phase) {
    return CODE_ROOTS;
  }
  if (phase === "phase-1") {
    return CODE_ROOTS.filter((root) => root.endsWith("phase-1-fundamentals"));
  }
  if (phase === "phase-2") {
    return CODE_ROOTS.filter((root) => root.endsWith("phase-2-rag"));
  }
  if (phase === "phase-3") {
    return CODE_ROOTS.filter((root) => root.endsWith("phase-3-frameworks"));
  }
  return CODE_ROOTS.filter((root) => root.endsWith("phase-4-advanced"));
}

export function isMarkdown(path: string): boolean {
  return extname(path).toLowerCase() === ".md";
}

export function isCodeFile(path: string): boolean {
  return [".py", ".ts", ".tsx", ".js", ".mjs"].includes(extname(path).toLowerCase());
}

export function titleFromMarkdown(text: string, path: string): string {
  const title = text.match(/^#\s+(.+)$/m)?.[1]?.trim();
  return title || basename(path);
}

export function titleFromCode(path: string): string {
  return basename(path);
}

export function scoreAndSnippet(text: string, query: string): { score: number; snippet: string } | null {
  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();
  const terms = lowerQuery.split(/\s+/).filter(Boolean);
  let firstIndex = lowerText.indexOf(lowerQuery);
  let score = firstIndex >= 0 ? 4 : 0;
  for (const term of terms) {
    const index = lowerText.indexOf(term);
    if (index >= 0) {
      score += 1;
      if (firstIndex < 0 || index < firstIndex) {
        firstIndex = index;
      }
    }
  }
  if (score === 0 || firstIndex < 0) {
    return null;
  }
  const start = Math.max(0, firstIndex - 80);
  const end = Math.min(text.length, firstIndex + query.length + 180);
  const snippet = text.slice(start, end).replace(/\s+/g, " ").trim();
  return { score, snippet };
}

export function sortHits<T extends { score: number; path: string }>(hits: T[]): T[] {
  return hits.sort((left, right) => right.score - left.score || left.path.localeCompare(right.path));
}

export function asJsonText(payload: unknown): { content: Array<{ type: "text"; text: string }> } {
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(payload, null, 2)
      }
    ]
  };
}
