import { clampLimit, assertNonEmptyQuery, toProjectRelative } from "../safety/path_guard.js";
import {
  docsRootsForPhase,
  isMarkdown,
  normalizePhase,
  phaseFromPath,
  readTextFile,
  scoreAndSnippet,
  sortHits,
  titleFromMarkdown,
  walkFiles,
  type SearchHit
} from "./common.js";

export interface SearchDocsArgs {
  query: string;
  phase?: string;
  limit?: number;
}

export interface SearchDocsResult {
  query: string;
  phase?: string;
  count: number;
  results: SearchHit[];
}

export async function searchDocs(args: SearchDocsArgs): Promise<SearchDocsResult> {
  const query = assertNonEmptyQuery(args.query);
  const phase = normalizePhase(args.phase);
  const limit = clampLimit(args.limit);
  const files = (
    await Promise.all(docsRootsForPhase(phase).map((root) => walkFiles(root, isMarkdown)))
  ).flat();

  const hits: SearchHit[] = [];
  for (const file of files) {
    const text = await readTextFile(file);
    const match = scoreAndSnippet(text, query);
    if (!match) {
      continue;
    }
    hits.push({
      path: toProjectRelative(file),
      phase: phaseFromPath(file),
      title: titleFromMarkdown(text, file),
      score: match.score,
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
