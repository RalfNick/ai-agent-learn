from __future__ import annotations

import hashlib
from typing import Sequence

from .models import Chunk, Document


class MarkdownChunker:
    def __init__(self, max_chars: int = 900, overlap_chars: int = 120) -> None:
        if max_chars < 80:
            raise ValueError("max_chars must be at least 80")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split_documents(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            for ordinal, content in enumerate(self._split_text(document.content)):
                chunks.append(
                    Chunk(
                        chunk_id=self._chunk_id(document.document_id, ordinal, content),
                        document_id=document.document_id,
                        path=document.path,
                        title=document.title,
                        content=content,
                        ordinal=ordinal,
                    )
                )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        if not normalized:
            return []
        if len(normalized) <= self.max_chars:
            return [normalized]

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + self.max_chars, len(normalized))
            if end < len(normalized):
                boundary = max(
                    normalized.rfind("\n\n", start, end),
                    normalized.rfind("\n#", start, end),
                    normalized.rfind("。", start, end),
                    normalized.rfind(".", start, end),
                )
                if boundary > start + self.max_chars // 3:
                    end = boundary + 1
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(0, end - self.overlap_chars)
        return chunks

    @staticmethod
    def _chunk_id(document_id: str, ordinal: int, content: str) -> str:
        digest = hashlib.sha1(f"{document_id}:{ordinal}:{content}".encode("utf-8")).hexdigest()
        return digest[:20]
