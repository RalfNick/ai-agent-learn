from __future__ import annotations

import re
from typing import Sequence

from knowledge.models import RetrievalResult

from .models import QASource


def context_score(results: Sequence[RetrievalResult]) -> float:
    if not results:
        return 0.0
    best_score = max(result.score for result in results)
    average_score = sum(result.score for result in results) / len(results)
    return round(0.7 * best_score + 0.3 * average_score, 6)


def results_to_sources(results: Sequence[RetrievalResult]) -> list[QASource]:
    return [
        QASource(
            source_id=result.chunk.chunk_id,
            title=result.chunk.title,
            path=result.chunk.path,
            score=result.score,
            snippet=snippet(result.chunk.content),
        )
        for result in results
    ]


def build_evidence_answer(question: str, results: Sequence[RetrievalResult]) -> str:
    evidence_lines = []
    for index, result in enumerate(results, start=1):
        sentence = best_sentence(result.chunk.content, question)
        evidence_lines.append(f"{index}. {sentence}（来源：{result.chunk.title}）")
    return "根据当前知识库资料，可以确认：\n" + "\n".join(evidence_lines)


def answer_is_evidence_supported(answer: str, sources: Sequence[QASource]) -> bool:
    if not answer.strip():
        return False
    source_titles = {source.title for source in sources}
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("根据当前知识库资料"):
            continue
        if not re.match(r"^\d+\.", stripped):
            continue
        if "（来源：" not in stripped:
            return False
        if not any(f"（来源：{title}）" in stripped for title in source_titles):
            return False
    return True


def best_sentence(content: str, question: str) -> str:
    sentences = candidate_evidence_lines(content)
    if not sentences:
        return snippet(remove_fenced_blocks(content))
    question_terms = set(simple_terms(question))
    exact_terms = exact_ascii_terms(question)
    ranked = sorted(
        sentences,
        key=lambda sentence: candidate_score(sentence, question_terms, exact_terms),
        reverse=True,
    )
    return ranked[0]


def candidate_evidence_lines(content: str) -> list[str]:
    cleaned_content = remove_fenced_blocks(content)
    candidates: list[str] = []
    for line in cleaned_content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        table_evidence = table_row_to_evidence(stripped)
        if table_evidence:
            candidates.append(table_evidence)
            continue
        for sentence in re.split(r"(?<=[。！？.!?])\s+", stripped):
            normalized = sentence.strip()
            if normalized and not is_markdown_noise(normalized):
                candidates.append(normalized)
    return candidates


def table_row_to_evidence(line: str) -> str | None:
    if not line.startswith("|") or not line.endswith("|"):
        return None
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    if len(cells) < 2:
        return None
    if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
        return None
    if set(cells[:2]) == {"能力", "为什么需要"}:
        return None
    if cells[0] in {"能力", "字段", "指标", "文件"}:
        return None
    return "：".join(cell for cell in cells if cell)


def candidate_score(
    sentence: str,
    question_terms: set[str],
    exact_terms: set[str],
) -> float:
    sentence_terms = set(simple_terms(sentence))
    overlap_score = len(question_terms.intersection(sentence_terms))
    exact_score = sum(1 for term in exact_terms if term in sentence.lower())
    return overlap_score + exact_score * 3.0


def exact_ascii_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
    }


def simple_terms(text: str) -> list[str]:
    terms = [token.lower() for token in re.findall(r"[\w\u4e00-\u9fff]+", text)]
    expanded = []
    for term in terms:
        expanded.append(term)
        chinese_chars = [char for char in term if "\u4e00" <= char <= "\u9fff"]
        expanded.extend(chinese_chars)
        expanded.extend(
            "".join(chinese_chars[index : index + 2])
            for index in range(len(chinese_chars) - 1)
        )
    return expanded


def is_markdown_noise(sentence: str) -> bool:
    stripped = sentence.strip()
    if stripped.startswith("!["):
        return True
    if stripped.startswith(("```", "|", "#", "-", "*", ">")):
        return True
    if stripped.startswith(("python ", "python3 ", "curl ", "npm ", "pip ", "cd ")):
        return True
    if stripped in {"text", "bash", "python", "json"}:
        return True
    if re.fullmatch(r"[A-Za-z]?[\]\"'`),;:]+", stripped):
        return True
    if stripped.endswith(("?", "？")):
        return True
    if "_" in stripped and len(re.findall(r"[A-Za-z_]{3,}", stripped)) >= 2:
        return True
    if re.match(r"^[|:\-\s]+$", stripped):
        return True
    return False


def remove_fenced_blocks(content: str) -> str:
    return re.sub(r"```.*?```", "", content, flags=re.DOTALL)


def snippet(content: str, limit: int = 260) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"
