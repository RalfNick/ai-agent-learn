import { readFile } from "node:fs/promises";
import { BENCHMARK_PHASES, BENCHMARK_SUMMARIES, type BenchmarkPhase } from "../config.js";
import { ensureInsideProject, toProjectRelative } from "../safety/path_guard.js";

export interface ReadBenchmarkSummaryArgs {
  phase?: string;
}

export interface BenchmarkSummary {
  phase: BenchmarkPhase;
  source: string;
  rows: Record<string, string>[];
}

export interface ReadBenchmarkSummaryResult {
  count: number;
  summaries: BenchmarkSummary[];
}

export async function readBenchmarkSummary(args: ReadBenchmarkSummaryArgs = {}): Promise<ReadBenchmarkSummaryResult> {
  const phases = normalizeBenchmarkPhases(args.phase);
  const summaries: BenchmarkSummary[] = [];
  for (const phase of phases) {
    const source = ensureInsideProject(BENCHMARK_SUMMARIES[phase]);
    const text = await readFile(source, "utf-8");
    summaries.push({
      phase,
      source: toProjectRelative(source),
      rows: parseCsv(text)
    });
  }
  return {
    count: summaries.length,
    summaries
  };
}

function normalizeBenchmarkPhases(phase?: string): BenchmarkPhase[] {
  if (!phase || phase === "all") {
    return [...BENCHMARK_PHASES];
  }
  if ((BENCHMARK_PHASES as string[]).includes(phase)) {
    return [phase as BenchmarkPhase];
  }
  throw new Error(`unsupported benchmark phase: ${phase}`);
}

function parseCsv(text: string): Record<string, string>[] {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    return [];
  }
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function splitCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (char === '"' && next === '"') {
      current += '"';
      index += 1;
      continue;
    }
    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  values.push(current);
  return values;
}
