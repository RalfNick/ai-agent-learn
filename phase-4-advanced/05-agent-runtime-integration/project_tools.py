from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


PHASES = {"phase-1", "phase-2", "phase-3", "phase-4"}
BENCHMARK_FILES = {
    "phase-2": "phase-2-rag/05-rag-benchmark/outputs/benchmark_summary.csv",
    "phase-3": "phase-3-frameworks/02-agentic-rag-langgraph/outputs/agentic_rag_summary.csv",
}
BLOCKED_PARTS = {"node_modules", "dist", "__pycache__", ".git", ".memory", ".pytest_cache"}
CODE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".mjs"}


@dataclass
class SearchHit:
    path: str
    title: str
    phase: str
    score: float
    snippet: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    query: str
    count: int
    results: list[SearchHit]
    phase: str | None = None


@dataclass
class BenchmarkSummary:
    phase: str
    source: str
    rows: list[dict[str, str]]


@dataclass
class BenchmarkResult:
    count: int
    summaries: list[BenchmarkSummary]


class ProjectToolset:
    """Read-only project tools mirroring the Phase4 MCP Server learning surface."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()

    def search_docs(self, query: str, phase: str | None = None, limit: int = 5) -> SearchResult:
        normalized_query = self._normalize_query(query)
        normalized_phase = self._normalize_phase(phase)
        limit = self._clamp_limit(limit)
        roots = [self.project_root / "docs" / normalized_phase] if normalized_phase else [self.project_root / "docs"]
        hits = self._search_files(normalized_query, roots, lambda path: path.suffix == ".md")
        results = sorted(hits, key=lambda item: (-item.score, item.path))[:limit]
        return SearchResult(query=normalized_query, phase=normalized_phase, count=len(results), results=results)

    def find_code_examples(self, query: str, phase: str | None = None, limit: int = 5) -> SearchResult:
        normalized_query = self._normalize_query(query)
        normalized_phase = self._normalize_phase(phase)
        limit = self._clamp_limit(limit)
        roots = self._code_roots(normalized_phase)
        hits = self._search_files(normalized_query, roots, lambda path: path.suffix in CODE_SUFFIXES)
        results = sorted(hits, key=lambda item: (-item.score, item.path))[:limit]
        return SearchResult(query=normalized_query, phase=normalized_phase, count=len(results), results=results)

    def read_benchmark_summary(self, phase: str | None = None) -> BenchmarkResult:
        phases = self._benchmark_phases(phase)
        summaries: list[BenchmarkSummary] = []

        for item in phases:
            path = self._ensure_inside_project(self.project_root / BENCHMARK_FILES[item])
            if not path.exists():
                summaries.append(BenchmarkSummary(phase=item, source=self._relative(path), rows=[]))
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
            summaries.append(BenchmarkSummary(phase=item, source=self._relative(path), rows=rows))

        return BenchmarkResult(count=len(summaries), summaries=summaries)

    def _search_files(
        self,
        query: str,
        roots: Iterable[Path],
        should_include: Callable[[Path], bool],
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for root in roots:
            root = self._ensure_inside_project(root)
            if not root.exists():
                continue
            for path in self._walk(root):
                if not should_include(path):
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                match = self._score_and_snippet(text, query)
                path_match = self._score_and_snippet(self._relative(path), query)
                selected = match or path_match
                if selected is None:
                    continue
                score, snippet = selected
                if path_match:
                    score += 2
                hits.append(
                    SearchHit(
                        path=self._relative(path),
                        title=self._title_for(path, text),
                        phase=self._phase_from_path(path),
                        score=score,
                        snippet=snippet,
                        metadata={"kind": "doc" if path.suffix == ".md" else "code"},
                    )
                )
        return hits

    def _walk(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if any(part in BLOCKED_PARTS for part in path.parts):
                continue
            if path.is_file():
                yield self._ensure_inside_project(path)

    def _score_and_snippet(self, text: str, query: str) -> tuple[float, str] | None:
        lower_text = text.lower()
        lower_query = query.lower()
        terms = [term for term in re.split(r"\s+", lower_query) if term]
        first_index = lower_text.find(lower_query)
        score = 4.0 if first_index >= 0 else 0.0

        for term in terms:
            index = lower_text.find(term)
            if index >= 0:
                score += 1.0
                if first_index < 0 or index < first_index:
                    first_index = index

        if score == 0 or first_index < 0:
            return None

        start = max(0, first_index - 80)
        end = min(len(text), first_index + len(query) + 180)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        return score, snippet

    def _normalize_query(self, query: str) -> str:
        normalized = query.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        if len(normalized) > 200:
            raise ValueError("query must be 200 characters or fewer")
        return normalized

    def _normalize_phase(self, phase: str | None) -> str | None:
        if phase is None:
            return None
        if phase not in PHASES:
            raise ValueError(f"unsupported phase: {phase}")
        return phase

    def _benchmark_phases(self, phase: str | None) -> list[str]:
        if phase is None or phase == "all":
            return list(BENCHMARK_FILES)
        if phase not in BENCHMARK_FILES:
            raise ValueError(f"unsupported benchmark phase: {phase}")
        return [phase]

    def _clamp_limit(self, limit: int) -> int:
        if not isinstance(limit, int) or limit < 1 or limit > 20:
            raise ValueError("limit must be an integer between 1 and 20")
        return limit

    def _code_roots(self, phase: str | None) -> list[Path]:
        mapping = {
            "phase-1": ["phase-1-fundamentals"],
            "phase-2": ["phase-2-rag"],
            "phase-3": ["phase-3-frameworks"],
            "phase-4": ["phase-4-advanced"],
        }
        names = mapping.get(phase, ["phase-1-fundamentals", "phase-2-rag", "phase-3-frameworks", "phase-4-advanced"])
        return [self.project_root / name for name in names]

    def _title_for(self, path: Path, text: str) -> str:
        if path.suffix == ".md":
            match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        return path.name

    def _phase_from_path(self, path: Path) -> str:
        rel = self._relative(path)
        match = re.search(r"(?:docs/)?(phase-[1-4])|phase-([1-4])-", rel)
        if not match:
            return "unknown"
        return match.group(1) or f"phase-{match.group(2)}"

    def _ensure_inside_project(self, path: Path) -> Path:
        absolute = path.resolve()
        if absolute != self.project_root and self.project_root not in absolute.parents:
            raise ValueError(f"path escapes project root: {path}")
        return absolute

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.project_root).as_posix()
