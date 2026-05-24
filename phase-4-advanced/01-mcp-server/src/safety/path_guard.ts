import { relative, resolve, sep } from "node:path";
import { CODE_ROOTS, DOCS_ROOT, PROJECT_ROOT } from "../config.js";

const ALLOWED_ROOTS = [DOCS_ROOT, ...CODE_ROOTS].map((item) => resolve(item));
const BLOCKED_PARTS = new Set(["node_modules", "dist", "__pycache__", ".git", ".ruff_cache", ".gradio"]);

export function assertNonEmptyQuery(query: string): string {
  const normalized = query.trim();
  if (!normalized) {
    throw new Error("query must not be empty");
  }
  if (normalized.length > 200) {
    throw new Error("query must be 200 characters or fewer");
  }
  return normalized;
}

export function ensureInsideProject(path: string): string {
  const absolute = resolve(path);
  const rel = relative(PROJECT_ROOT, absolute);
  if (rel.startsWith("..") || rel === "" || rel.includes(`..${sep}`)) {
    throw new Error(`path escapes project root: ${path}`);
  }
  return absolute;
}

export function ensureInsideAllowedRoots(path: string): string {
  const absolute = ensureInsideProject(path);
  const allowed = ALLOWED_ROOTS.some((root) => {
    const rel = relative(root, absolute);
    return rel === "" || (!rel.startsWith("..") && !rel.includes(`..${sep}`));
  });
  if (!allowed) {
    throw new Error(`path is not in the read-only MCP allowlist: ${path}`);
  }
  return absolute;
}

export function toProjectRelative(path: string): string {
  return relative(PROJECT_ROOT, ensureInsideProject(path)).split(sep).join("/");
}

export function isBlockedPath(path: string): boolean {
  const parts = toProjectRelative(path).split("/");
  return parts.some((part) => BLOCKED_PARTS.has(part));
}

export function clampLimit(limit: number | undefined, fallback = 5, max = 20): number {
  if (limit === undefined) {
    return fallback;
  }
  if (!Number.isInteger(limit) || limit < 1 || limit > max) {
    throw new Error(`limit must be an integer between 1 and ${max}`);
  }
  return limit;
}
