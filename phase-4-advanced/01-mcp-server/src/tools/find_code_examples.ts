import { extname } from "node:path";
import { clampLimit, assertNonEmptyQuery, toProjectRelative } from "../safety/path_guard.js";
import {
  codeRootsForPhase,
  isCodeFile,
  normalizePhase,
  phaseFromPath,
  readTextFile,
  scoreAndSnippet,
  sortHits,
  titleFromCode,
  walkFiles,
  type SearchHit
} from "./common.js";

export interface FindCodeExamplesArgs {
  query: string;
  phase?: string;
  limit?: number;
}

export interface CodeHit extends SearchHit {
  language: string;
}

export interface FindCodeExamplesResult {
  query: string;
  phase?: string;
  count: number;
  results: CodeHit[];
}

const LANGUAGE_BY_EXT: Record<string, string> = {
  ".py": "python",
  ".ts": "typescript",
  ".tsx": "typescript",
  ".js": "javascript",
  ".mjs": "javascript"
};

export async function findCodeExamples(args: FindCodeExamplesArgs): Promise<FindCodeExamplesResult> {
  const query = assertNonEmptyQuery(args.query);
  const phase = normalizePhase(args.phase);
  const limit = clampLimit(args.limit);
  const files = (
    await Promise.all(codeRootsForPhase(phase).map((root) => walkFiles(root, isCodeFile)))
  ).flat();

  const hits: CodeHit[] = [];
  for (const file of files) {
    const text = await readTextFile(file);
    const fileNameMatch = scoreAndSnippet(toProjectRelative(file), query);
    const contentMatch = scoreAndSnippet(text, query);
    const match = contentMatch ?? fileNameMatch;
    if (!match) {
      continue;
    }
    hits.push({
      path: toProjectRelative(file),
      phase: phaseFromPath(file),
      title: titleFromCode(file),
      language: LANGUAGE_BY_EXT[extname(file).toLowerCase()] ?? "text",
      score: match.score + (fileNameMatch ? 2 : 0),
      snippet: match.snippet
    });
  }

  const results = sortHits(hits).slice(0, limit);
  return {
    query,
    phase,
    count: results.length,
    results
  };
}
