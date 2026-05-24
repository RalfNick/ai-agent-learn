import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

export const PROJECT_ROOT = resolve(here, "../../..");

export const DOCS_ROOT = resolve(PROJECT_ROOT, "docs");

export const CODE_ROOTS = [
  "phase-1-fundamentals",
  "phase-2-rag",
  "phase-3-frameworks",
  "phase-4-advanced"
].map((item) => resolve(PROJECT_ROOT, item));

export const BENCHMARK_SUMMARIES = {
  "phase-2": resolve(PROJECT_ROOT, "phase-2-rag/05-rag-benchmark/outputs/benchmark_summary.csv"),
  "phase-3": resolve(PROJECT_ROOT, "phase-3-frameworks/02-agentic-rag-langgraph/outputs/agentic_rag_summary.csv")
} as const;

export type Phase = "phase-1" | "phase-2" | "phase-3" | "phase-4";
export type BenchmarkPhase = keyof typeof BENCHMARK_SUMMARIES;

export const PHASES: Phase[] = ["phase-1", "phase-2", "phase-3", "phase-4"];
export const BENCHMARK_PHASES: BenchmarkPhase[] = ["phase-2", "phase-3"];
