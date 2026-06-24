from __future__ import annotations

import json
import math
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Sequence

from .chunking import MarkdownChunker
from .loaders import load_documents
from .models import Chunk, IndexStats, RetrievalResult


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", flags=re.UNICODE)


class LocalKnowledgeIndex:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._chunk_tokens = [Counter(_tokenize(chunk.content)) for chunk in self._chunks]
        self._chunk_vectors = [_hashed_vector(tokens) for tokens in self._chunk_tokens]

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> "LocalKnowledgeIndex":
        return cls(chunks)

    def search(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        query_tokens = Counter(_tokenize(normalized_query))
        query_vector = _hashed_vector(query_tokens)
        results: list[RetrievalResult] = []

        for chunk, chunk_tokens, chunk_vector in zip(
            self._chunks, self._chunk_tokens, self._chunk_vectors
        ):
            lexical_score = _lexical_score(query_tokens, chunk_tokens)
            vector_score = _cosine_similarity(query_vector, chunk_vector)
            score = 0.65 * lexical_score + 0.35 * vector_score
            if score > 0:
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=round(score, 6),
                        lexical_score=round(lexical_score, 6),
                        vector_score=round(vector_score, 6),
                    )
                )

        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def stats(self) -> IndexStats:
        document_ids = {chunk.document_id for chunk in self._chunks}
        token_count = sum(sum(tokens.values()) for tokens in self._chunk_tokens)
        return IndexStats(
            document_count=len(document_ids),
            chunk_count=len(self._chunks),
            token_count=token_count,
        )

    def save(self, path: Path | str) -> None:
        index_path = Path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "stats": self.stats().to_dict(),
            "chunks": [chunk.to_dict() for chunk in self._chunks],
        }
        index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "LocalKnowledgeIndex":
        index_path = Path(path)
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        return cls(chunks)


def build_index_from_paths(
    paths: Sequence[Path | str],
    max_chars: int = 900,
    overlap_chars: int = 120,
    extensions: set[str] | None = None,
) -> LocalKnowledgeIndex:
    documents = load_documents(paths, extensions=extensions)
    chunks = MarkdownChunker(max_chars=max_chars, overlap_chars=overlap_chars).split_documents(
        documents
    )
    return LocalKnowledgeIndex.from_chunks(chunks)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(text):
        token = raw_token.lower()
        tokens.append(token)
        chinese_chars = [char for char in token if "\u4e00" <= char <= "\u9fff"]
        if chinese_chars:
            tokens.extend(chinese_chars)
            tokens.extend(
                "".join(chinese_chars[index : index + 2])
                for index in range(len(chinese_chars) - 1)
            )
    return tokens


def _lexical_score(query_tokens: Counter[str], chunk_tokens: Counter[str]) -> float:
    if not query_tokens:
        return 0.0
    overlap = sum(min(count, chunk_tokens.get(token, 0)) for token, count in query_tokens.items())
    return overlap / sum(query_tokens.values())


def _hashed_vector(tokens: Counter[str], dimensions: int = 128) -> dict[int, float]:
    vector: dict[int, float] = {}
    for token, count in tokens.items():
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % dimensions
        vector[bucket] = vector.get(bucket, 0.0) + float(count)
    return vector


def _cosine_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
